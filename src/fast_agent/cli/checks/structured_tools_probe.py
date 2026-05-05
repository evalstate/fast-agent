from __future__ import annotations

import json
import random
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Literal

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.tool_agent import ToolAgent
from fast_agent.core import Core
from fast_agent.llm.model_factory import ModelFactory
from fast_agent.llm.request_params import RequestParams
from fast_agent.llm.structured_schema import (
    validate_json_instance,
    validate_json_schema_definition,
)

StructuredToolPolicy = Literal["auto", "always", "defer", "no_tools"]
StructuredProbeMode = Literal["direct", "tools"]

PROBE_SCHEMA = validate_json_schema_definition(
    {
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
)

DIRECT_PROBE_SCHEMA = validate_json_schema_definition(
    {
        "type": "object",
        "properties": {
            "city": {"type": "string"},
            "condition": {"type": "string"},
            "temperature_c": {"type": "integer"},
            "summary": {"type": "string"},
        },
        "required": ["city", "condition", "temperature_c", "summary"],
        "additionalProperties": False,
    }
)


@dataclass(slots=True)
class ProbeResult:
    mode: StructuredProbeMode
    model: str
    resolved_model: str | None
    provider: str | None
    json_mode: str | None
    structured_tool_policy: StructuredToolPolicy | None
    passed: bool
    tool_calls: int
    final_json_valid: bool
    matched_tool_payload: bool
    matched_direct_payload: bool
    stop_reason: str | None
    response_text: str | None
    parsed: dict[str, Any] | None
    error: str | None = None


def _build_tools_prompt() -> str:
    return (
        "You must call the `get_probe_payload` tool before answering. "
        "The payload changes every run, so do not guess. "
        #        "After the tool result arrives, return only JSON that matches the required schema."
    )


def _build_direct_prompt() -> str:
    return (
        "Return a concise JSON weather report for Paris. Use city Paris, "
        "condition Sunny, temperature_c 21, and a short summary."
    )


def _llm_metadata(agent: ToolAgent) -> tuple[str | None, str | None, str | None]:
    if agent.llm is None:
        return None, None, None
    resolved_model = agent.llm.resolved_model
    return (
        resolved_model.wire_model_name if resolved_model is not None else None,
        agent.llm.provider.config_name,
        resolved_model.json_mode if resolved_model is not None else None,
    )


async def _probe_direct_model(core: Core, model: str) -> ProbeResult:
    agent = ToolAgent(
        AgentConfig(name="direct-structured-probe", model=model),
        tools=[],
        context=core.context,
    )

    try:
        await agent.attach_llm(ModelFactory.create_factory(model))
        parsed, response = await agent.structured_schema(
            _build_direct_prompt(),
            DIRECT_PROBE_SCHEMA,
            RequestParams(use_history=False, maxTokens=1024),
        )
        if not isinstance(parsed, dict):
            raise ValueError(f"structured response was not a JSON object: {type(parsed).__name__}")

        validate_json_instance(parsed, DIRECT_PROBE_SCHEMA)
        matched = (
            parsed.get("city") == "Paris"
            and parsed.get("condition") == "Sunny"
            and parsed.get("temperature_c") == 21
        )
        if not matched:
            raise ValueError("weather response did not match the requested payload")

        resolved_model, provider, json_mode = _llm_metadata(agent)
        response_text = response.last_text()
        stop_reason = response.stop_reason.value if response.stop_reason is not None else None
        return ProbeResult(
            mode="direct",
            model=model,
            resolved_model=resolved_model,
            provider=provider,
            json_mode=json_mode,
            structured_tool_policy=None,
            passed=True,
            tool_calls=0,
            final_json_valid=True,
            matched_tool_payload=False,
            matched_direct_payload=True,
            stop_reason=stop_reason,
            response_text=response_text,
            parsed=parsed,
        )
    except Exception as exc:
        resolved_model, provider, json_mode = _llm_metadata(agent)
        return ProbeResult(
            mode="direct",
            model=model,
            resolved_model=resolved_model,
            provider=provider,
            json_mode=json_mode,
            structured_tool_policy=None,
            passed=False,
            tool_calls=0,
            final_json_valid=False,
            matched_tool_payload=False,
            matched_direct_payload=False,
            stop_reason=None,
            response_text=None,
            parsed=None,
            error=str(exc),
        )
    finally:
        with suppress(Exception):
            await agent.shutdown()


async def _probe_tools_model(
    core: Core,
    model: str,
    *,
    structured_tool_policy: StructuredToolPolicy,
) -> ProbeResult:
    probe_id = f"probe-{random.SystemRandom().randint(100_000, 999_999)}"
    magic_number = random.SystemRandom().randint(10_000_000, 99_999_999)
    tool_call_count = 0

    async def get_probe_payload() -> dict[str, str | int]:
        """Return the current probe payload required for the final structured answer.

        Call this tool to obtain the authoritative probe_id and magic_number for
        this run. Do not guess or invent these values.
        """
        nonlocal tool_call_count
        tool_call_count += 1
        return {
            "probe_id": probe_id,
            "magic_number": magic_number,
            "tool_name": "get_probe_payload",
        }

    agent = ToolAgent(
        AgentConfig(name="tools-structured-probe", model=model),
        tools=[get_probe_payload],
        context=core.context,
    )

    request_params = RequestParams(
        use_history=False,
        structured_schema=PROBE_SCHEMA,
        structured_tool_policy=structured_tool_policy,
        maxTokens=1024,
        max_iterations=4,
    )

    try:
        await agent.attach_llm(ModelFactory.create_factory(model))
        response = await agent.generate(_build_tools_prompt(), request_params=request_params)
        resolved_model, provider, json_mode = _llm_metadata(agent)
        response_text = response.last_text()
        if response_text is None:
            raise ValueError("assistant response did not include text content")

        parsed = json.loads(response_text)
        if not isinstance(parsed, dict):
            raise ValueError(f"structured response was not a JSON object: {type(parsed).__name__}")

        validate_json_instance(parsed, PROBE_SCHEMA)

        if tool_call_count < 1:
            raise ValueError("tool was not called")
        if parsed.get("probe_id") != probe_id:
            raise ValueError("probe_id did not match the tool result")
        if parsed.get("magic_number") != magic_number:
            raise ValueError("magic_number did not match the tool result")
        if parsed.get("tool_name") != "get_probe_payload":
            raise ValueError("tool_name did not match the expected tool")

        stop_reason = response.stop_reason.value if response.stop_reason is not None else None
        return ProbeResult(
            mode="tools",
            model=model,
            resolved_model=resolved_model,
            provider=provider,
            json_mode=json_mode,
            structured_tool_policy=structured_tool_policy,
            passed=True,
            tool_calls=tool_call_count,
            final_json_valid=True,
            matched_tool_payload=True,
            matched_direct_payload=False,
            stop_reason=stop_reason,
            response_text=response_text,
            parsed=parsed,
        )
    except Exception as exc:
        resolved_model, provider, json_mode = _llm_metadata(agent)
        return ProbeResult(
            mode="tools",
            model=model,
            resolved_model=resolved_model,
            provider=provider,
            json_mode=json_mode,
            structured_tool_policy=structured_tool_policy,
            passed=False,
            tool_calls=tool_call_count,
            final_json_valid=False,
            matched_tool_payload=False,
            matched_direct_payload=False,
            stop_reason=None,
            response_text=None,
            parsed=None,
            error=str(exc),
        )
    finally:
        with suppress(Exception):
            await agent.shutdown()

def _print_text_summary(results: list[ProbeResult]) -> None:
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        provider = result.provider or "unknown"
        policy = result.structured_tool_policy or "-"
        details = (
            f"mode={result.mode} policy={policy} "
            f"tool_calls={result.tool_calls} stop_reason={result.stop_reason or '-'}"
        )
        print(f"{status:4} {result.model:28} provider={provider:18} {details}")
        if result.error:
            print(f"      error: {result.error}")

    passed = sum(1 for result in results if result.passed)
    print(f"\nSummary: {passed}/{len(results)} passed")


async def run_probe(
    models: list[str],
    *,
    structured_tool_policy: StructuredToolPolicy,
    mode: StructuredProbeMode = "tools",
) -> list[ProbeResult]:
    core = Core()
    await core.initialize()
    try:
        results: list[ProbeResult] = []
        for model in models:
            if mode == "direct":
                results.append(await _probe_direct_model(core, model))
            else:
                results.append(
                    await _probe_tools_model(
                        core,
                        model,
                        structured_tool_policy=structured_tool_policy,
                    )
                )
        return results
    finally:
        await core.cleanup()


async def run_probe_suite(
    models: list[str],
    *,
    structured_tool_policy: StructuredToolPolicy,
    modes: list[StructuredProbeMode],
) -> list[ProbeResult]:
    core = Core()
    await core.initialize()
    try:
        results: list[ProbeResult] = []
        for model in models:
            for mode in modes:
                if mode == "direct":
                    results.append(await _probe_direct_model(core, model))
                else:
                    results.append(
                        await _probe_tools_model(
                            core,
                            model,
                            structured_tool_policy=structured_tool_policy,
                        )
                    )
        return results
    finally:
        await core.cleanup()
