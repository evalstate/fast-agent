#!/usr/bin/env python3

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Literal

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.tool_agent import ToolAgent
from fast_agent.core import Core
from fast_agent.llm.model_database import ModelDatabase, ModelParameters
from fast_agent.llm.model_factory import ModelFactory
from fast_agent.llm.provider_types import Provider
from fast_agent.types import RequestParams

if TYPE_CHECKING:
    from fast_agent.llm.request_params import StructuredToolPolicy

JsonMode = Literal["schema", "object", "none"]

SCHEMA = {
    "type": "object",
    "properties": {
        "probe_id": {"type": "string"},
        "status": {"type": "string"},
    },
    "required": ["probe_id", "status"],
    "additionalProperties": False,
}

TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "probe_id": {"type": "string"},
        "magic_number": {"type": "integer"},
        "tool_name": {"type": "string"},
        "summary": {"type": "string"},
    },
    "required": ["probe_id", "magic_number", "tool_name", "summary"],
    "additionalProperties": False,
}


@dataclass(slots=True)
class ModeProbe:
    mode: JsonMode
    passed: bool
    parsed: dict[str, Any] | None
    error: str | None


@dataclass(slots=True)
class ToolPolicyProbe:
    policy: StructuredToolPolicy
    passed: bool
    tool_calls: int
    final_json_valid: bool
    matched_tool_payload: bool
    error: str | None


@dataclass(slots=True)
class SupportProbeResult:
    model: str
    resolved_model: str
    provider: str
    mode_probes: list[ModeProbe]
    recommended_json_mode: JsonMode | None
    always: ToolPolicyProbe | None
    defer: ToolPolicyProbe | None
    recommended_policy: str | None


def _base_model_name(model_name: str) -> str:
    if ":" in model_name:
        return model_name.split(":", 1)[0]
    return model_name


def _mode_value(mode: JsonMode) -> str | None:
    return None if mode == "none" else mode


def _runtime_params(model_name: str, mode: JsonMode) -> ModelParameters:
    base_model = _base_model_name(model_name)
    existing = ModelDatabase.get_model_params(base_model, provider=Provider.HUGGINGFACE)
    if existing is not None:
        return existing.model_copy(update={"json_mode": _mode_value(mode)})
    return ModelParameters(
        context_window=262_144,
        max_output_tokens=16_384,
        tokenizes=list(ModelDatabase.TEXT_ONLY),
        json_mode=_mode_value(mode),
        default_provider=Provider.HUGGINGFACE,
    )


async def _with_mode[T](model_name: str, mode: JsonMode, probe):
    base_model = _base_model_name(model_name)
    ModelDatabase.register_runtime_model_params(base_model, _runtime_params(model_name, mode))
    try:
        return await probe()
    finally:
        ModelDatabase.unregister_runtime_model_params(base_model)


async def _probe_mode(core: Core, model: str, mode: JsonMode) -> ModeProbe:
    cfg = ModelFactory.parse_model_string(model)
    probe_id = f"mode-{random.SystemRandom().randint(100_000, 999_999)}"

    async def run() -> ModeProbe:
        agent = ToolAgent(AgentConfig(name="structured-mode-probe", model=model), [], core.context)
        await agent.attach_llm(ModelFactory.create_factory(model))
        try:
            parsed, _ = await agent.structured_schema(
                f'Return JSON with probe_id="{probe_id}" and status="ok".',
                SCHEMA,
                RequestParams(use_history=False, maxTokens=768),
            )
            passed = parsed == {"probe_id": probe_id, "status": "ok"}
            return ModeProbe(
                mode=mode,
                passed=passed,
                parsed=parsed if isinstance(parsed, dict) else None,
                error=None if passed else "parsed JSON did not match expected payload",
            )
        except Exception as exc:
            return ModeProbe(
                mode=mode, passed=False, parsed=None, error=f"{type(exc).__name__}: {exc}"
            )
        finally:
            await agent.shutdown()

    return await _with_mode(cfg.model_name, mode, run)


async def _probe_policy(
    core: Core,
    model: str,
    mode: JsonMode,
    policy: StructuredToolPolicy,
) -> ToolPolicyProbe:
    cfg = ModelFactory.parse_model_string(model)
    probe_id = f"tool-{random.SystemRandom().randint(100_000, 999_999)}"
    magic_number = random.SystemRandom().randint(10_000_000, 99_999_999)
    tool_calls = 0

    async def get_probe_payload() -> dict[str, str | int]:
        """Return the current probe payload required for the final structured answer."""
        nonlocal tool_calls
        tool_calls += 1
        return {
            "probe_id": probe_id,
            "magic_number": magic_number,
            "tool_name": "get_probe_payload",
        }

    async def run() -> ToolPolicyProbe:
        agent = ToolAgent(
            AgentConfig(name="structured-tools-matrix-probe", model=model),
            [get_probe_payload],
            core.context,
        )
        await agent.attach_llm(ModelFactory.create_factory(model))
        try:
            parsed, _ = await agent.structured_schema(
                "You must call get_probe_payload before answering. "
                "The payload changes every run, so do not guess.",
                TOOL_SCHEMA,
                RequestParams(
                    use_history=False,
                    maxTokens=1024,
                    max_iterations=4,
                    structured_tool_policy=policy,
                ),
            )
            final_json_valid = isinstance(parsed, dict)
            matched = (
                final_json_valid
                and parsed.get("probe_id") == probe_id
                and parsed.get("magic_number") == magic_number
                and parsed.get("tool_name") == "get_probe_payload"
            )
            passed = tool_calls > 0 and final_json_valid and matched
            return ToolPolicyProbe(
                policy=policy,
                passed=passed,
                tool_calls=tool_calls,
                final_json_valid=final_json_valid,
                matched_tool_payload=matched,
                error=None if passed else "tool was not called or final JSON did not match payload",
            )
        except Exception as exc:
            return ToolPolicyProbe(
                policy=policy,
                passed=False,
                tool_calls=tool_calls,
                final_json_valid=False,
                matched_tool_payload=False,
                error=f"{type(exc).__name__}: {exc}",
            )
        finally:
            await agent.shutdown()

    return await _with_mode(cfg.model_name, mode, run)


async def probe_model(core: Core, model: str) -> SupportProbeResult:
    cfg = ModelFactory.parse_model_string(model)
    mode_probes: list[ModeProbe] = []
    recommended_mode: JsonMode | None = None
    for mode in ("schema", "object", "none"):
        result = await _probe_mode(core, model, mode)
        mode_probes.append(result)
        if result.passed and recommended_mode is None:
            recommended_mode = mode
            break

    always: ToolPolicyProbe | None = None
    defer: ToolPolicyProbe | None = None
    recommended_policy: str | None = None
    if recommended_mode is not None:
        always = await _probe_policy(core, model, recommended_mode, "always")
        if always.passed:
            recommended_policy = "always"
        else:
            defer = await _probe_policy(core, model, recommended_mode, "defer")
            recommended_policy = "defer" if defer.passed else "no_tools"

    return SupportProbeResult(
        model=model,
        resolved_model=cfg.model_name,
        provider=cfg.provider.config_name,
        mode_probes=mode_probes,
        recommended_json_mode=recommended_mode,
        always=always,
        defer=defer,
        recommended_policy=recommended_policy,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe structured-output support matrix.")
    parser.add_argument("--models", required=True, help="Comma-separated model aliases/specs.")
    parser.add_argument("--json", action="store_true", help="Emit JSON only.")
    return parser.parse_args()


def _print_table(results: list[SupportProbeResult]) -> None:
    print("| Model | Resolved | Mode | Always | Defer | Recommended |")
    print("|---|---|---:|---:|---:|---|")
    for result in results:
        always = "-" if result.always is None else "pass" if result.always.passed else "fail"
        defer = "-" if result.defer is None else "pass" if result.defer.passed else "fail"
        print(
            f"| `{result.model}` | `{result.resolved_model}` | "
            f"`{result.recommended_json_mode}` | {always} | {defer} | "
            f"`{result.recommended_policy}` |"
        )


async def _run() -> int:
    args = _parse_args()
    models = [model.strip() for model in args.models.split(",") if model.strip()]
    core = Core()
    await core.initialize()
    try:
        results = [await probe_model(core, model) for model in models]
    finally:
        await core.cleanup()

    if args.json:
        print(json.dumps([asdict(result) for result in results], indent=2))
    else:
        _print_table(results)
        print()
        print(json.dumps([asdict(result) for result in results], indent=2))
    return 0 if all(result.recommended_json_mode for result in results) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(_run()))
