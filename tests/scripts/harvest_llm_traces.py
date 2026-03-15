from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal, NotRequired, TypedDict, cast

os.environ.setdefault("FAST_AGENT_LLM_TRACE", "1")

from mcp import Tool

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.llm_agent import LlmAgent
from fast_agent.constants import (
    ANTHROPIC_CITATIONS_CHANNEL,
    ANTHROPIC_SERVER_TOOLS_CHANNEL,
)
from fast_agent.core import Core
from fast_agent.llm.model_factory import ModelFactory
from fast_agent.llm.request_params import RequestParams
from fast_agent.types.llm_stop_reason import LlmStopReason

if TYPE_CHECKING:
    from fast_agent.mcp.prompt_message_extended import PromptMessageExtended

REPO_ROOT = Path(__file__).resolve().parents[2]
TRACE_FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures" / "llm_traces"
TRACE_MATRIX_PATH = TRACE_FIXTURES_DIR / "manifests" / "trace_matrix.json"
DEFAULT_CONFIG_PATH = REPO_ROOT / "tests" / "e2e" / "llm" / "fastagent.config.yaml"
STREAM_DEBUG_DIR = REPO_ROOT / "stream-debug"

ScenarioKind = Literal["plain_text", "tool_use", "web_search"]


class ScenarioSpecDict(TypedDict):
    kind: ScenarioKind
    prompt_file: str
    max_tokens: NotRequired[int]
    toolset: NotRequired[str]
    enable_web_search: NotRequired[bool]


class TraceMatrixDict(TypedDict):
    scenarios: dict[str, ScenarioSpecDict]
    targets: dict[str, list[str]]


@dataclass(frozen=True, slots=True)
class ScenarioSpec:
    name: str
    kind: ScenarioKind
    prompt_file: Path
    max_tokens: int | None = None
    toolset: str | None = None
    enable_web_search: bool = False


@dataclass(slots=True)
class HarvestCapture:
    stream_events: list[dict[str, Any]]
    tool_events: list[dict[str, Any]]


WEATHER_TOOL = Tool(
    name="weather",
    description="Check the weather in a city",
    inputSchema={
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "City to check",
            }
        },
        "required": ["city"],
    },
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Harvest real provider traces for replay fixtures.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="fast-agent config file to use for local runs.",
    )
    parser.add_argument(
        "--matrix",
        type=Path,
        default=TRACE_MATRIX_PATH,
        help="Trace matrix JSON file.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=TRACE_FIXTURES_DIR / "raw",
        help="Root directory for raw captured runs.",
    )
    parser.add_argument(
        "--model",
        "--models",
        dest="models",
        action="append",
        default=[],
        help="Limit capture to the given model. Repeatable.",
    )
    parser.add_argument(
        "--scenario",
        "--scenarios",
        dest="scenarios",
        action="append",
        default=[],
        help="Limit capture to the given scenario. Repeatable.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep harvesting after a scenario fails its expectations.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List the configured model/scenario matrix and exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the runs that would be executed without calling providers.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        help="Override the scenario max token budget for one-off retries.",
    )
    return parser.parse_args()


def _load_trace_matrix(path: Path) -> tuple[dict[str, ScenarioSpec], dict[str, list[str]]]:
    payload = cast("TraceMatrixDict", json.loads(path.read_text(encoding="utf-8")))
    scenarios = {
        name: ScenarioSpec(
            name=name,
            kind=spec["kind"],
            prompt_file=(path.parent.parent / spec["prompt_file"]).resolve(),
            max_tokens=spec.get("max_tokens"),
            toolset=spec.get("toolset"),
            enable_web_search=spec.get("enable_web_search", False),
        )
        for name, spec in payload["scenarios"].items()
    }
    return scenarios, payload["targets"]


def _model_slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", model).strip("-")


def _now_tag() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return _jsonable(model_dump(mode="json"))
        except TypeError:
            return _jsonable(model_dump())
        except Exception:
            return str(value)
    model_dump_json = getattr(value, "model_dump_json", None)
    if callable(model_dump_json):
        try:
            return json.loads(model_dump_json())
        except Exception:
            return str(value)
    if hasattr(value, "__dict__"):
        try:
            return {
                str(key): _jsonable(item)
                for key, item in vars(value).items()
                if not str(key).startswith("_")
            }
        except Exception:
            return str(value)
    return str(value)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")


def _append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(payload), sort_keys=True) + "\n")


def _snapshot_stream_debug() -> set[Path]:
    if not STREAM_DEBUG_DIR.exists():
        return set()
    return {path.resolve() for path in STREAM_DEBUG_DIR.iterdir() if path.is_file()}


def _collect_new_stream_debug(before: set[Path]) -> list[Path]:
    if not STREAM_DEBUG_DIR.exists():
        return []
    after = {path.resolve() for path in STREAM_DEBUG_DIR.iterdir() if path.is_file()}
    return sorted(after - before)


def _move_trace_files(paths: list[Path], destination: Path) -> list[str]:
    moved: list[str] = []
    destination.mkdir(parents=True, exist_ok=True)
    for path in paths:
        target = destination / path.name
        shutil.move(str(path), str(target))
        moved.append(target.name)
    return moved


def _install_google_trace_capture(llm: Any, run_dir: Path) -> Callable[[], None]:
    if not hasattr(llm, "_consume_google_stream") or not hasattr(llm, "_stream_generate_content"):
        return lambda: None

    original_consume = llm._consume_google_stream
    original_stream = llm._stream_generate_content
    request_path = run_dir / "google_stream_request.json"
    chunk_path = run_dir / "google_stream_chunks.jsonl"

    async def wrapped_stream_generate_content(
        *,
        model: str,
        contents: list[Any],
        config: Any,
        client: Any,
    ) -> Any:
        _write_json(
            request_path,
            {
                "model": model,
                "contents": contents,
                "config": config,
            },
        )
        return await original_stream(model=model, contents=contents, config=config, client=client)

    async def wrapped_consume(response_stream: Any, *, model: str) -> Any:
        async def logged_stream() -> Any:
            async for chunk in response_stream:
                _append_jsonl(chunk_path, chunk)
                yield chunk

        return await original_consume(logged_stream(), model=model)

    llm._stream_generate_content = wrapped_stream_generate_content
    llm._consume_google_stream = wrapped_consume

    def restore() -> None:
        llm._stream_generate_content = original_stream
        llm._consume_google_stream = original_consume

    return restore


def _capture_listeners(llm: Any, run_dir: Path) -> tuple[HarvestCapture, Callable[[], None]]:
    capture = HarvestCapture(stream_events=[], tool_events=[])

    def on_stream(chunk: Any) -> None:
        event = {
            "text": getattr(chunk, "text", ""),
            "is_reasoning": bool(getattr(chunk, "is_reasoning", False)),
        }
        capture.stream_events.append(event)
        _append_jsonl(run_dir / "listener_stream.jsonl", event)

    def on_tool(event_type: str, payload: dict[str, Any] | None = None) -> None:
        event = {
            "event_type": event_type,
            "payload": payload or {},
        }
        capture.tool_events.append(event)
        _append_jsonl(run_dir / "listener_tools.jsonl", event)

    remove_stream = llm.add_stream_listener(on_stream)
    remove_tool = llm.add_tool_stream_listener(on_tool)

    def cleanup() -> None:
        remove_stream()
        remove_tool()

    return capture, cleanup


def _set_web_search(llm: Any, enabled: bool) -> None:
    setter = getattr(llm, "set_web_search_enabled", None)
    if callable(setter):
        setter(enabled)

    fetch_setter = getattr(llm, "set_web_fetch_enabled", None)
    if callable(fetch_setter):
        try:
            fetch_setter(False)
        except Exception:
            pass


def _supports_toolset(name: str | None) -> list[Tool] | None:
    if name is None:
        return None
    if name == "weather":
        return [WEATHER_TOOL]
    raise ValueError(f"Unknown toolset: {name}")


def _has_tool_event(capture: HarvestCapture, *, tool_name: str | None = None) -> bool:
    for event in capture.tool_events:
        payload = cast("dict[str, Any]", event.get("payload") or {})
        if tool_name is None:
            return True
        if payload.get("tool_name") == tool_name:
            return True
    return False


def _validate_result(
    *,
    scenario: ScenarioSpec,
    result: PromptMessageExtended,
    capture: HarvestCapture,
) -> list[str]:
    issues: list[str] = []

    if scenario.kind == "plain_text":
        if result.stop_reason is not LlmStopReason.END_TURN:
            issues.append(f"expected end_turn, got {result.stop_reason}")
        if not (result.last_text() or "").strip():
            issues.append("expected non-empty assistant text")
        return issues

    if scenario.kind == "tool_use":
        if result.stop_reason is not LlmStopReason.TOOL_USE:
            issues.append(f"expected tool_use, got {result.stop_reason}")
        if not result.tool_calls:
            issues.append("expected tool_calls in final result")
        if not _has_tool_event(capture):
            issues.append("expected tool stream events")
        return issues

    if scenario.kind == "web_search":
        channels = result.channels or {}
        has_web_search_event = _has_tool_event(capture, tool_name="web_search")
        has_server_tool_channel = bool(channels.get(ANTHROPIC_SERVER_TOOLS_CHANNEL))
        has_citations = bool(channels.get(ANTHROPIC_CITATIONS_CHANNEL))
        if not (has_web_search_event or has_server_tool_channel or has_citations):
            issues.append("expected web-search evidence in tool events or channels")
        if not (result.last_text() or "").strip():
            issues.append("expected non-empty assistant text")
        return issues

    issues.append(f"unsupported scenario kind: {scenario.kind}")
    return issues


async def _run_scenario(
    *,
    config_path: Path,
    requested_model: str,
    scenario: ScenarioSpec,
    run_dir: Path,
    max_tokens_override: int | None,
) -> dict[str, Any]:
    trace_snapshot = _snapshot_stream_debug()
    prompt = scenario.prompt_file.read_text(encoding="utf-8").strip()
    started_at = datetime.now(UTC)
    metadata: dict[str, Any] = {
        "requested_model": requested_model,
        "scenario": scenario.name,
        "prompt_file": str(scenario.prompt_file.relative_to(REPO_ROOT)),
        "started_at": started_at.isoformat(),
        "fast_agent_llm_trace": os.environ.get("FAST_AGENT_LLM_TRACE", ""),
        "scenario_max_tokens": scenario.max_tokens,
        "max_tokens_override": max_tokens_override,
    }

    async with Core(settings=config_path).run() as core:
        agent = LlmAgent(AgentConfig("trace-harvest"), core.context)
        await agent.attach_llm(ModelFactory.create_factory(requested_model))

        llm = agent.llm
        assert llm is not None

        metadata["provider"] = llm.provider.config_name
        metadata["llm_class"] = llm.__class__.__name__
        metadata["resolved_model"] = llm.default_request_params.model

        google_restore = _install_google_trace_capture(llm, run_dir)
        capture, remove_listeners = _capture_listeners(llm, run_dir)

        try:
            if scenario.enable_web_search:
                _set_web_search(llm, True)

            tools = _supports_toolset(scenario.toolset)
            effective_max_tokens = (
                max_tokens_override if max_tokens_override is not None else scenario.max_tokens
            )
            request_params = (
                RequestParams(maxTokens=effective_max_tokens)
                if effective_max_tokens is not None
                else None
            )
            result = await agent.generate(
                prompt,
                request_params=request_params,
                tools=tools,
            )
        finally:
            remove_listeners()
            google_restore()

    _write_json(run_dir / "result.json", result.model_dump(mode="json"))

    trace_files = _move_trace_files(_collect_new_stream_debug(trace_snapshot), run_dir)
    validation_issues = _validate_result(scenario=scenario, result=result, capture=capture)

    metadata.update(
        {
            "completed_at": datetime.now(UTC).isoformat(),
            "stop_reason": result.stop_reason.value if result.stop_reason else None,
            "has_tool_calls": bool(result.tool_calls),
            "stream_event_count": len(capture.stream_events),
            "tool_event_count": len(capture.tool_events),
            "trace_files": trace_files,
            "validation_issues": validation_issues,
            "status": "ok" if not validation_issues else "validation_failed",
        }
    )
    _write_json(run_dir / "meta.json", metadata)
    return metadata


def _selected_runs(
    *,
    scenarios: dict[str, ScenarioSpec],
    targets: dict[str, list[str]],
    requested_models: list[str],
    requested_scenarios: list[str],
) -> list[tuple[str, ScenarioSpec]]:
    models = requested_models or list(targets)
    selected: list[tuple[str, ScenarioSpec]] = []
    for model in models:
        scenario_names = targets.get(model)
        if scenario_names is None:
            if not requested_scenarios:
                raise KeyError(
                    "Model not found in trace matrix and no explicit scenarios were provided: "
                    f"{model}"
                )
            scenario_names = requested_scenarios
        for scenario_name in scenario_names:
            if scenario_name not in scenarios:
                raise KeyError(f"Scenario not found in trace matrix: {scenario_name}")
            if requested_scenarios and scenario_name not in requested_scenarios:
                continue
            selected.append((model, scenarios[scenario_name]))
    return selected


def _print_matrix(scenarios: dict[str, ScenarioSpec], targets: dict[str, list[str]]) -> None:
    print("Scenarios:")
    for scenario in scenarios.values():
        print(
            f"  - {scenario.name}: kind={scenario.kind} prompt={scenario.prompt_file.relative_to(REPO_ROOT)}"
        )
    print("\nTargets:")
    for model, scenario_names in targets.items():
        joined = ", ".join(scenario_names)
        print(f"  - {model}: {joined}")


async def _amain() -> int:
    os.chdir(REPO_ROOT)
    args = _parse_args()
    scenarios, targets = _load_trace_matrix(args.matrix.resolve())

    if args.list:
        _print_matrix(scenarios, targets)
        return 0

    selected = _selected_runs(
        scenarios=scenarios,
        targets=targets,
        requested_models=args.models,
        requested_scenarios=args.scenarios,
    )
    if not selected:
        print("No matching runs selected.")
        return 1

    print(f"Repository root: {REPO_ROOT}")
    print(f"Config path: {args.config.resolve()}")
    print(f"Output root: {args.output_root.resolve()}")
    print(f"FAST_AGENT_LLM_TRACE={os.environ.get('FAST_AGENT_LLM_TRACE', '')}")

    failures = 0
    for requested_model, scenario in selected:
        model_slug = _model_slug(requested_model)
        run_dir = args.output_root.resolve() / model_slug / scenario.name / _now_tag()
        print(f"\n=== {requested_model} :: {scenario.name} ===")
        print(f"-> {run_dir.relative_to(REPO_ROOT)}")
        if args.dry_run:
            continue
        run_dir.mkdir(parents=True, exist_ok=False)
        try:
            metadata = await _run_scenario(
                config_path=args.config.resolve(),
                requested_model=requested_model,
                scenario=scenario,
                run_dir=run_dir,
                max_tokens_override=args.max_tokens,
            )
        except Exception as exc:
            failures += 1
            error_metadata = {
                "requested_model": requested_model,
                "scenario": scenario.name,
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "completed_at": datetime.now(UTC).isoformat(),
            }
            _write_json(run_dir / "meta.json", error_metadata)
            print(f"ERROR: {type(exc).__name__}: {exc}")
            if not args.continue_on_error:
                return 1
            continue

        if metadata["status"] != "ok":
            failures += 1
            print(f"FAILED: {', '.join(cast('list[str]', metadata['validation_issues']))}")
            if not args.continue_on_error:
                return 1
        else:
            print(
                "OK:"
                f" provider={metadata['provider']}"
                f" resolved_model={metadata['resolved_model']}"
                f" traces={len(cast('list[str]', metadata['trace_files']))}"
            )

    return 1 if failures else 0


def main() -> None:
    raise SystemExit(asyncio.run(_amain()))


if __name__ == "__main__":
    main()
