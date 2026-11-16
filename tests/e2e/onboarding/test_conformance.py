"""Short known-answer probes through the same harness used by applications."""

from __future__ import annotations

import base64
import json
import re
from typing import TYPE_CHECKING

import pytest
from mcp_types import BlobResourceContents, EmbeddedResource
from pydantic import BaseModel

from fast_agent.llm.model_mime_support import tokenizes_support_mime
from fast_agent.mcp.prompt import Prompt
from fast_agent.ptc import PTC_TOOL_NAME
from fast_agent.tools.tool_surface import ToolExposure, resolve_tool_surface
from fast_agent.types.llm_stop_reason import LlmStopReason

from .conftest import HERE, Notes
from .harness import harness, reasoning_chars

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = [pytest.mark.e2e, pytest.mark.asyncio]


def contains_number(text: str | None, value: int) -> bool:
    return bool(re.search(rf"(?<!\d){value}(?!\d)", text or ""))


@pytest.mark.onboarding("text-stream", "usage")
async def test_text_stream_and_usage(evidence: Notes, workspace: Path) -> None:
    async with harness(evidence, workspace) as run:
        result = await run.agent.generate("What is 2 + 2? Answer with one digit.")
        assert result.stop_reason is LlmStopReason.END_TURN
        text = result.last_text()
        assert text and text.strip()
        assert run.chunks > 0
        usage = run.agent.usage_accumulator
        assert usage is not None
        assert (usage.summary.prompt.total or 0) > 0
        assert (usage.summary.completion.total or 0) > 0
        assert usage.context_usage_percentage is not None
        assert usage.context_usage_percentage > 0


@pytest.mark.onboarding("reasoning-efforts")
async def test_reasoning(evidence: Notes, workspace: Path, effort: str) -> None:
    if effort == "not-applicable":
        pytest.skip("Catalog has no reasoning effort spec")
    evidence.note("reasoning-efforts", requested=effort)
    async with harness(evidence, workspace, effort=effort) as run:
        result = await run.agent.generate("What is 17 + 25? Answer only the integer.")
        assert result.stop_reason is LlmStopReason.END_TURN
        assert contains_number(result.last_text(), 42)
        assert run.agent.llm is not None
        effective = run.agent.llm.reasoning_effort
        if effective and (effective.value is False or effective.value in ("none", 0)):
            reasoning_observed = reasoning_chars(result) > 0 or run.reasoning_chunks > 0
            evidence.note("reasoning-efforts", reasoning_observed_when_disabled=reasoning_observed)
            if reasoning_observed:
                evidence.note(
                    "reasoning-efforts",
                    warning="Reasoning observed at off/none/zero; consult the provider contract for hard versus soft disable semantics",
                )
    details = evidence.state.tests["reasoning-efforts"][evidence.nodeid].details
    if not details["wire_reasoning"]:
        evidence.note(
            "reasoning-efforts",
            warning="Provider did not expose request capture; wire value unverified",
        )


@pytest.mark.onboarding("tool-loop")
@pytest.mark.parametrize("parallel", [False, True], ids=["single", "two-independent"])
async def test_function_tool(evidence: Notes, workspace: Path, parallel: bool) -> None:
    async with harness(evidence, workspace, tools=True) as run:
        prompt = (
            "Call lookup_value for alpha and beta independently, in parallel if possible. Report both integers."
            if parallel
            else "Call lookup_value for alpha. Report its integer."
        )
        result = await run.agent.generate(prompt)
        assert result.stop_reason is LlmStopReason.END_TURN
        assert "alpha" in run.calls
        assert contains_number(result.last_text(), 137)
        if parallel:
            assert "beta" in run.calls
            assert contains_number(result.last_text(), 286)
        history = run.agent.message_history
        assert any(m.tool_calls for m in history)
        assert any(m.tool_results for m in history)


@pytest.mark.onboarding("tool-loop")
async def test_shell_tool(evidence: Notes, workspace: Path) -> None:
    async with harness(evidence, workspace, shell=True) as run:
        result = await run.agent.generate(
            "Use the shell to run exactly: printf 'onboard-%s' 731. Report its output."
        )
        assert result.stop_reason is LlmStopReason.END_TURN
        assert "onboard-731" in (result.last_text() or "")
        history = run.agent.message_history
        assert any(m.tool_calls for m in history)
        outputs = [r for m in history for r in (m.tool_results or {}).values()]
        assert outputs and not any(r.is_error for r in outputs)
        assert any("onboard-731" in r.model_dump_json() for r in outputs)


@pytest.mark.onboarding("multi-turn")
async def test_history(evidence: Notes, workspace: Path) -> None:
    if evidence.state.track == "lite":
        pytest.skip("multi-turn is full track only")
    async with harness(evidence, workspace, tools=True) as run:
        first = await run.agent.generate("Remember the secret integer 59. Acknowledge briefly.")
        assert first.stop_reason is LlmStopReason.END_TURN
        middle = await run.agent.generate("Call lookup_value for alpha and report the integer.")
        assert middle.stop_reason is LlmStopReason.END_TURN
        assert "alpha" in run.calls
        last = await run.agent.generate(
            "Add the secret integer from my first turn to the tool value. Give the sum."
        )
        assert last.stop_reason is LlmStopReason.END_TURN
        assert contains_number(last.last_text(), 196)


class Answer(BaseModel):
    answer: int


@pytest.mark.onboarding("structured")
@pytest.mark.parametrize("with_tools", [False, True], ids=["schema", "schema-with-tool"])
async def test_structured(evidence: Notes, workspace: Path, with_tools: bool) -> None:
    resolved = evidence.state.resolved
    assert resolved is not None
    if resolved.json_mode is None:
        pytest.skip("Catalog declares no JSON output mode")
    async with harness(evidence, workspace, tools=with_tools) as run:
        assert run.agent.llm is not None
        policy = run.agent.llm.resolve_structured_tool_policy(run.agent.llm.get_request_params())
        evidence.note("structured", json_mode=resolved.json_mode, tool_policy=policy)
        needs_tool = with_tools and policy != "no_tools"
        prompt = (
            "Call lookup_value for alpha. Return its integer as answer."
            if needs_tool
            else "What is 2 + 2? Return the integer as answer."
        )
        if with_tools:
            data, result = await run.agent.structured_schema(
                [Prompt.user(prompt)], schema=Answer.model_json_schema()
            )
            try:
                response_json = json.loads(result.last_text() or "")
            except ValueError:
                evidence.note("structured", response_json_kind="not-json")
            else:
                evidence.note(
                    "structured",
                    response_json_kind=type(response_json).__name__,
                    response_json_keys=sorted(response_json)[:10]
                    if isinstance(response_json, dict)
                    else [],
                )
            evidence.note(
                "structured", schema_valid=data is not None, local_tool_calls=len(run.calls)
            )
            parsed = Answer.model_validate(data)
        else:
            parsed, result = await run.agent.structured([Prompt.user(prompt)], model=Answer)
        assert isinstance(parsed, Answer)
        assert parsed.answer == (137 if needs_tool else 4)
        assert result.stop_reason is LlmStopReason.END_TURN
        if needs_tool:
            assert "alpha" in run.calls
        elif with_tools:
            assert not run.calls


@pytest.mark.onboarding("media")
@pytest.mark.parametrize("kind", ["image", "pdf", "unsupported"])
async def test_media(evidence: Notes, workspace: Path, kind: str) -> None:
    resolved = evidence.state.resolved
    assert resolved is not None and resolved.model_params is not None
    mime = {
        "image": "image/png",
        "pdf": "application/pdf",
        "unsupported": "application/x-onboarding-opaque",
    }[kind]
    supported = tokenizes_support_mime(resolved.model_params.tokenizes, mime)
    evidence.note("media", mime=mime, catalog_supported=supported)
    if kind == "unsupported":
        assert not supported
        attachment = EmbeddedResource(
            type="resource",
            resource=BlobResourceContents(
                uri="file:///onboarding.opaque",
                mime_type=mime,
                blob=base64.b64encode(b"opaque fixture").decode(),
            ),
        )
        prompt = Prompt.user("If the attachment cannot be read, say so briefly.", attachment)
    else:
        path = HERE.parent / "multimodal" / ("image.png" if kind == "image" else "sample.pdf")
        question = (
            "What is the user name in this image? Be brief."
            if kind == "image"
            else "Name the product or company in this PDF. Be brief."
        )
        prompt = Prompt.user(
            question if supported else "Can you read this attachment? Be brief.", path
        )
    async with harness(evidence, workspace) as run:
        result = await run.agent.generate(prompt)
        evidence.note(
            "media",
            response_stop=result.stop_reason.value if result.stop_reason else None,
            response_chars=len(result.last_text() or ""),
        )
        assert result.stop_reason in (
            (LlmStopReason.END_TURN,)
            if supported
            else (LlmStopReason.END_TURN, LlmStopReason.ERROR)
        )
        text = (result.last_text() or "").lower()
        assert text.strip()
        if supported:
            assert (
                ("evalstate" in text)
                if kind == "image"
                else ("fast-agent" in text or "llmindset" in text)
            )


@pytest.mark.onboarding("ptc/uptake")
async def test_ptc(evidence: Notes, workspace: Path) -> None:
    if evidence.state.track == "lite":
        pytest.skip("ptc/uptake is full track only")
    async with harness(evidence, workspace, tools=True, ptc=True) as run:
        surface = await resolve_tool_surface(run.agent)
        exposed = [t.python_name for t in surface.exposed(ToolExposure.PROGRAMMATIC)]
        evidence.note(
            "ptc/uptake",
            exposed_functions=[name for name in exposed],
            exposed_stubs=[t.stub for t in surface.exposed(ToolExposure.PROGRAMMATIC)],
        )
        result = await run.agent.generate(
            "Use Python code to call lookup_value for alpha and beta, add the results, and print the sum. Report the sum."
        )
        history = run.agent.message_history
        code_ids = {
            key
            for m in history
            for key, call in (m.tool_calls or {}).items()
            if call.params.name == PTC_TOOL_NAME
        }
        nested: list[str] = []
        for message in history:
            for key, tool_result in (message.tool_results or {}).items():
                if key not in code_ids:
                    continue
                metadata = tool_result.meta or {}
                calls = metadata.get("fast-agent-ptc-calls", [])
                assert isinstance(calls, list)
                for call in calls:
                    assert isinstance(call, dict) and isinstance(call["tool"], str)
                    nested.append(call["tool"])
        evidence.note("ptc/uptake", nested_functions=[name for name in nested])
        assert result.stop_reason is LlmStopReason.END_TURN
        assert code_ids and nested
        assert {"alpha", "beta"} <= set(run.calls)
        assert contains_number(result.last_text(), 423)
