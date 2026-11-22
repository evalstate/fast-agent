"""Real harness setup and bounded observations, never provider substitutes."""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING
from urllib.parse import parse_qsl, urlencode

from fastmcp.tools import FunctionTool
from pydantic import JsonValue, TypeAdapter

from fast_agent import FastAgent
from fast_agent.agents.tool_agent import ToolAgent
from fast_agent.constants import REASONING
from fast_agent.llm.request_params import RequestParams
from fast_agent.mcp.helpers.content_helpers import get_text

from .conftest import HERE, Notes

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable
    from pathlib import Path

    from fast_agent.interfaces import AgentProtocol
    from fast_agent.llm.stream_types import StreamChunk
    from fast_agent.mcp.prompt_message_extended import PromptMessageExtended

JSON_OBJECT = TypeAdapter(dict[str, JsonValue])


def selection(model: str, **overrides: str) -> str:
    base, _, query = model.partition("?")
    return base + "?" + urlencode({**dict(parse_qsl(query)), **overrides})


@dataclass
class Observation:
    agent: AgentProtocol
    chunks: int = 0
    reasoning_chunks: int = 0
    calls: list[str] = field(default_factory=list)

    def stream(self, chunk: StreamChunk) -> None:
        if chunk.text:
            self.chunks += 1
            self.reasoning_chunks += int(chunk.is_reasoning)


def reasoning_chars(message: PromptMessageExtended) -> int:
    return sum(len(get_text(block) or "") for block in (message.channels or {}).get(REASONING, []))


def wire_reasoning(workspace: Path) -> list[dict[str, JsonValue]]:
    """Read only whitelisted reasoning fields; never persist requests or transcripts."""
    observations: list[dict[str, JsonValue]] = []
    keys = {
        "reasoning_effort",
        "thinking",
        "reasoning",
        "thinking_config",
        "thinkingConfig",
        "output_config",
        "enable_thinking",
    }
    scalar_keys = {
        "type",
        "effort",
        "budget_tokens",
        "thinking_budget",
        "thinkingBudget",
        "thinking_level",
        "thinkingLevel",
        "includeThoughts",
        "enabled",
    }

    def extract(payload: dict[str, JsonValue]) -> dict[str, JsonValue]:
        found: dict[str, JsonValue] = {}
        for key, value in payload.items():
            if key in keys:
                if isinstance(value, dict):
                    found[key] = {k: v for k, v in value.items() if k in scalar_keys}
                elif value is None or isinstance(value, (str, int, float, bool)):
                    found[key] = value
            elif key in {"arguments", "extra_body", "config", "generation_config"}:
                if isinstance(value, dict):
                    found.update(extract(value))
        return found

    for path in sorted((workspace / "stream-debug").glob("*request.json")):
        payload = JSON_OBJECT.validate_json(path.read_text())
        observations.append(extract(payload))
    return observations


@asynccontextmanager
async def harness(
    evidence: Notes,
    workspace: Path,
    *,
    effort: str | None = None,
    tools: bool = False,
    shell: bool = False,
    ptc: bool = False,
) -> AsyncGenerator[Observation]:
    state = evidence.state
    assert state.resolved is not None
    overrides = {"ptc": "on" if ptc else "off"}
    if effort is not None and effort != "default":
        overrides["reasoning"] = effort
    model = selection(state.model, **overrides)
    fast = FastAgent(
        "onboarding",
        config_path=str(HERE / "fast-agent.yaml"),
        parse_cli_args=False,
        quiet=True,
        isolated=True,
        workspace=workspace,
        home=workspace,
    )
    calls: list[str] = []
    ptc_lookup: FunctionTool | None = None
    if tools:

        @fast.tool
        def lookup_value(key: str) -> int:
            """Read the current integer for key alpha or beta from the local source."""
            calls.append(key)
            return {"alpha": 137, "beta": 286}[key]

        if ptc:
            # Plain @fast.tool results are text-only. Keep this arithmetic probe
            # about PTC uptake, not implicit conversion of scalar text to integers.
            ptc_lookup = FunctionTool.from_function(lookup_value)
            assert ptc_lookup.output_schema is not None
            ptc_lookup.meta = {"fast_agent": {"ptc": {"returns": ptc_lookup.output_schema}}}

    output_limit = 2048
    if effort and effort.isdecimal():
        output_limit = max(output_limit, int(effort) + 1024)
    params = state.resolved.model_params
    if params:
        output_limit = min(output_limit, params.max_output_tokens)

    @fast.agent(
        "probe",
        model=model,
        instruction="Be concise. Use tools when requested. Do not guess tool results.",
        shell=shell,
        cwd=workspace,
        skills=[],
        request_params=RequestParams(max_tokens=output_limit, max_iterations=5),
    )
    async def registered() -> None:
        pass

    async with fast.run() as app:
        agent = app["probe"]
        if ptc_lookup is not None:
            assert isinstance(agent, ToolAgent)
            agent.add_tool(ptc_lookup)
        assert agent.llm is not None
        observation = Observation(agent=agent, calls=calls)
        remove: Callable[[], None] = agent.llm.add_stream_listener(observation.stream)
        try:
            yield observation
        finally:
            remove()
            history = agent.message_history
            batches = [len(m.tool_calls) for m in history if m.tool_calls]
            tool_calls = [
                (key, call) for m in history for key, call in (m.tool_calls or {}).items()
            ]
            results = {
                key: result for m in history for key, result in (m.tool_results or {}).items()
            }
            setting = agent.llm.reasoning_effort
            spec = agent.llm.reasoning_effort_spec
            effective = setting or (spec.default if spec else None)
            usage = agent.usage_accumulator
            details = JSON_OBJECT.validate_python(
                {
                    "selected_model": model,
                    "max_tokens": output_limit,
                    "configured_reasoning": asdict(setting) if setting else None,
                    "effective_reasoning": asdict(effective) if effective else None,
                    "effective_reasoning_source": "configured" if setting else "catalog_default",
                    "wire_reasoning": wire_reasoning(workspace),
                    "stream_chunks": observation.chunks,
                    "reasoning_chunks": observation.reasoning_chunks,
                    "reasoning_chars": sum(reasoning_chars(m) for m in history),
                    "stop_reasons": [m.stop_reason.value for m in history if m.stop_reason],
                    "tool_call_count": len(tool_calls),
                    "tool_names": [call.params.name for _, call in tool_calls],
                    "tool_results": len(results),
                    "tool_errors": sum(bool(r.is_error) for r in results.values()),
                    "parallel_calls_observed": any(n > 1 for n in batches),
                    "usage": usage.get_summary() if usage else None,
                }
            )
            for check, tests in state.tests.items():
                if evidence.nodeid in tests:
                    evidence.note(check, **details)
            if usage:
                state.usage.extend(usage.turns)
