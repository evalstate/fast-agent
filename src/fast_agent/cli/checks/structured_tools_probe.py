from __future__ import annotations

import json
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

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
StructuredProbeMode = Literal["direct", "pydantic", "tools"]
StructuredProbeCase = Literal["json_schema", "pydantic"]

ORDER_ID = "ORD-7291"
PROBE_MAX_TOKENS = 2048

ORDER_REPORT_SCHEMA = validate_json_schema_definition(
    {
        "$defs": {
            "line_item": {
                "type": "object",
                "properties": {
                    "sku": {"type": "string", "description": "Inventory SKU."},
                    "quantity": {"type": "integer", "minimum": 1},
                    "unit_price_usd": {"type": "number", "minimum": 0},
                },
                "required": ["sku", "quantity", "unit_price_usd"],
                "additionalProperties": False,
            },
            "fulfillment": {
                "type": "object",
                "properties": {
                    "carrier": {"type": "string", "enum": ["DHL", "UPS", "FedEx", "LocalCourier"]},
                    "priority": {"type": "string", "enum": ["standard", "expedite"]},
                    "eta_days": {"type": "integer", "minimum": 1, "maximum": 14},
                },
                "required": ["carrier", "priority", "eta_days"],
                "additionalProperties": False,
            },
        },
        "type": "object",
        "properties": {
            "order_id": {"type": "string"},
            "customer_tier": {"type": "string", "enum": ["standard", "plus", "enterprise"]},
            "destination_city": {"type": "string"},
            "line_items": {
                "type": "array",
                "minItems": 2,
                "maxItems": 4,
                "items": {"$ref": "#/$defs/line_item"},
            },
            "fulfillment": {"$ref": "#/$defs/fulfillment"},
            "risk_flags": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": ["address_check", "inventory_watch", "weather_delay"],
                },
            },
            "total_usd": {"type": "number", "minimum": 0},
            "ready_to_ship": {"type": "boolean"},
            "summary": {"type": "string"},
        },
        "required": [
            "order_id",
            "customer_tier",
            "destination_city",
            "line_items",
            "fulfillment",
            "risk_flags",
            "total_usd",
            "ready_to_ship",
            "summary",
        ],
        "additionalProperties": False,
    }
)

PROBE_SCHEMA = ORDER_REPORT_SCHEMA
DIRECT_PROBE_SCHEMA = ORDER_REPORT_SCHEMA


class ProbeLineItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sku: str = Field(description="Inventory SKU")
    quantity: int = Field(ge=1)
    unit_price_usd: float = Field(ge=0)


class ProbeFulfillment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    carrier: Literal["DHL", "UPS", "FedEx", "LocalCourier"]
    priority: Literal["standard", "expedite"]
    eta_days: int = Field(ge=1, le=14)


class ProbeOrderSummary(BaseModel):
    """Medium-complexity structured-output probe model."""

    model_config = ConfigDict(extra="forbid")

    order_id: str
    customer_tier: Literal["standard", "plus", "enterprise"]
    destination_city: str
    line_items: list[ProbeLineItem] = Field(min_length=2, max_length=4)
    fulfillment: ProbeFulfillment
    risk_flags: list[Literal["address_check", "inventory_watch", "weather_delay"]]
    total_usd: float = Field(ge=0)
    ready_to_ship: bool
    summary: str


@dataclass(slots=True)
class ProbeResult:
    mode: StructuredProbeMode
    case: StructuredProbeCase
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


def _base_order_report(
    *, summary: str = "Paris order is ready for expedited DHL shipping."
) -> dict[str, Any]:
    return {
        "order_id": ORDER_ID,
        "customer_tier": "plus",
        "destination_city": "Paris",
        "line_items": [
            {"sku": "notebook", "quantity": 2, "unit_price_usd": 12.5},
            {"sku": "pen", "quantity": 5, "unit_price_usd": 1.2},
        ],
        "fulfillment": {"carrier": "DHL", "priority": "expedite", "eta_days": 3},
        "risk_flags": ["inventory_watch"],
        "total_usd": 31.0,
        "ready_to_ship": True,
        "summary": summary,
    }


def _build_tools_prompt() -> str:
    return (
        f"Use the `get_order_readiness` tool to look up order {ORDER_ID}. "
        "Then return one JSON order readiness report using the tool result. "
        "Preserve order_id, customer_tier, destination_city, line_items, fulfillment, "
        "risk_flags, total_usd, and ready_to_ship exactly as returned by the tool. "
        "Add only a concise summary string."
    )


def _build_direct_prompt() -> str:
    return (
        "Return exactly one JSON order readiness report with this data: "
        f"order_id {ORDER_ID}; customer_tier plus; destination_city Paris; "
        "line_items notebook quantity 2 unit_price_usd 12.5 and pen quantity 5 "
        "unit_price_usd 1.2; fulfillment carrier DHL priority expedite eta_days 3; "
        "risk_flags inventory_watch; total_usd 31.0; ready_to_ship true; "
        "summary Paris order is ready for expedited DHL shipping."
    )


def _build_pydantic_prompt() -> str:
    return (
        "Create an order summary object for validation. Use order_id ORD-7291, "
        "customer_tier plus, destination_city Paris, two line items "
        "(notebook x2 at 12.5 USD and pen x5 at 1.2 USD), DHL expedited "
        "fulfillment with eta_days 3, risk_flags [inventory_watch], total_usd 31.0, "
        "ready_to_ship true, and a concise summary."
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


def _matches_order_report(parsed: dict[str, Any], expected: dict[str, Any]) -> bool:
    return (
        parsed.get("order_id") == expected["order_id"]
        and parsed.get("customer_tier") == expected["customer_tier"]
        and parsed.get("destination_city") == expected["destination_city"]
        and parsed.get("line_items") == expected["line_items"]
        and parsed.get("fulfillment") == expected["fulfillment"]
        and parsed.get("risk_flags") == expected["risk_flags"]
        and parsed.get("total_usd") == expected["total_usd"]
        and parsed.get("ready_to_ship") == expected["ready_to_ship"]
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
            RequestParams(use_history=False, maxTokens=PROBE_MAX_TOKENS),
        )
        if not isinstance(parsed, dict):
            raise ValueError(f"structured response was not a JSON object: {type(parsed).__name__}")

        validate_json_instance(parsed, DIRECT_PROBE_SCHEMA)
        if not _matches_order_report(parsed, _base_order_report()):
            raise ValueError("order report did not match the requested payload")

        resolved_model, provider, json_mode = _llm_metadata(agent)
        response_text = response.last_text()
        stop_reason = response.stop_reason.value if response.stop_reason is not None else None
        return ProbeResult(
            mode="direct",
            case="json_schema",
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
            case="json_schema",
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


async def _probe_pydantic_model(core: Core, model: str) -> ProbeResult:
    agent = ToolAgent(
        AgentConfig(name="pydantic-structured-probe", model=model),
        tools=[],
        context=core.context,
    )

    try:
        await agent.attach_llm(ModelFactory.create_factory(model))
        result, response = await agent.structured(
            _build_pydantic_prompt(),
            ProbeOrderSummary,
            RequestParams(use_history=False, maxTokens=PROBE_MAX_TOKENS),
        )
        if result is None:
            raise ValueError("structured response did not validate as ProbeOrderSummary")

        parsed = result.model_dump(mode="json")
        validate_json_instance(parsed, ProbeOrderSummary.model_json_schema())
        if not _matches_order_report(parsed, _base_order_report()):
            raise ValueError("Pydantic order summary did not match the requested payload")

        resolved_model, provider, json_mode = _llm_metadata(agent)
        response_text = response.last_text()
        stop_reason = response.stop_reason.value if response.stop_reason is not None else None
        return ProbeResult(
            mode="pydantic",
            case="pydantic",
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
            mode="pydantic",
            case="pydantic",
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
    tool_call_count = 0
    tool_payload = _base_order_report(summary="")

    async def get_order_readiness(order_id: str) -> dict[str, Any]:
        """Return a fictional order readiness report.

        Use this read-only helper when an order needs current structured
        fulfillment fields before producing a final report.
        """
        nonlocal tool_call_count
        tool_call_count += 1
        if order_id != ORDER_ID:
            return {
                **tool_payload,
                "order_id": order_id,
                "ready_to_ship": False,
                "risk_flags": ["address_check"],
            }
        return tool_payload

    agent = ToolAgent(
        AgentConfig(name="tools-structured-probe", model=model),
        tools=[get_order_readiness],
        context=core.context,
    )

    request_params = RequestParams(
        use_history=False,
        structured_schema=PROBE_SCHEMA,
        structured_tool_policy=structured_tool_policy,
        maxTokens=PROBE_MAX_TOKENS,
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
        if not _matches_order_report(parsed, tool_payload):
            raise ValueError("order report did not match the tool result")

        stop_reason = response.stop_reason.value if response.stop_reason is not None else None
        return ProbeResult(
            mode="tools",
            case="json_schema",
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
            case="json_schema",
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
            f"mode={result.mode} case={result.case} policy={policy} "
            f"tool_calls={result.tool_calls} stop_reason={result.stop_reason or '-'}"
        )
        print(f"{status:4} {result.model:28} provider={provider:18} {details}")
        if result.error:
            print(f"      error: {result.error}")

    passed = sum(1 for result in results if result.passed)
    print(f"\nSummary: {passed}/{len(results)} passed")


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
                elif mode == "pydantic":
                    results.append(await _probe_pydantic_model(core, model))
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
