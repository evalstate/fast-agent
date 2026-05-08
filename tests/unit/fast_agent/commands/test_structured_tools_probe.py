from dataclasses import asdict

from fast_agent.cli.checks.structured_tools_probe import (
    ORDER_REPORT_SCHEMA,
    ProbeOrderSummary,
    ProbeResult,
    _base_order_report,
    _build_direct_prompt,
    _build_pydantic_prompt,
    _build_tools_prompt,
    _matches_order_report,
)
from fast_agent.llm.structured_schema import validate_json_instance


def test_probe_schema_is_medium_complexity_and_validates_sample_payload() -> None:
    payload = _base_order_report()

    validate_json_instance(payload, ORDER_REPORT_SCHEMA)

    assert "$defs" in ORDER_REPORT_SCHEMA
    assert "line_items" in ORDER_REPORT_SCHEMA["properties"]
    assert ORDER_REPORT_SCHEMA["properties"]["line_items"]["items"] == {"$ref": "#/$defs/line_item"}
    assert ORDER_REPORT_SCHEMA["properties"]["fulfillment"] == {"$ref": "#/$defs/fulfillment"}


def test_probe_pydantic_model_is_reasonably_complex_and_forbids_extra_fields() -> None:
    payload = _base_order_report()

    result = ProbeOrderSummary.model_validate(payload)
    dumped = result.model_dump(mode="json")

    assert dumped == payload
    assert ProbeOrderSummary.model_config["extra"] == "forbid"
    schema = ProbeOrderSummary.model_json_schema()
    assert "$defs" in schema
    assert schema["additionalProperties"] is False


def test_probe_prompts_exercise_same_order_payload() -> None:
    for prompt in (_build_direct_prompt(), _build_pydantic_prompt(), _build_tools_prompt()):
        assert "ORD-7291" in prompt

    for prompt in (_build_direct_prompt(), _build_pydantic_prompt()):
        assert "Paris" in prompt


def test_matches_order_report_ignores_summary_but_checks_structural_fields() -> None:
    expected = _base_order_report(summary="expected summary")
    actual = _base_order_report(summary="provider generated summary")

    assert _matches_order_report(actual, expected)

    actual["fulfillment"] = {**actual["fulfillment"], "eta_days": 5}
    assert not _matches_order_report(actual, expected)


def test_probe_result_serializes_new_case_field() -> None:
    result = ProbeResult(
        mode="pydantic",
        case="pydantic",
        model="example.model",
        resolved_model="wire-model",
        provider="example",
        json_mode="schema",
        structured_tool_policy=None,
        passed=True,
        tool_calls=0,
        final_json_valid=True,
        matched_tool_payload=False,
        matched_direct_payload=True,
        stop_reason="end_turn",
        response_text=None,
        parsed=_base_order_report(),
    )

    serialized = asdict(result)
    assert serialized["mode"] == "pydantic"
    assert serialized["case"] == "pydantic"
