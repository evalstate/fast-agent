"""One-shot CLI execution helpers."""

from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING, Any

import typer
from pydantic import BaseModel

if TYPE_CHECKING:
    from fast_agent.cli.runtime.run_request import AgentRunRequest
    from fast_agent.core.harness import HarnessSession
    from fast_agent.llm.structured_schema import StructuredSchemaSource
    from fast_agent.types import PromptMessageExtended, StructuredToolPolicy


async def structured_call(
    agent_obj: Any,
    prompt: Any,
    schema_source: "StructuredSchemaSource",
    structured_tool_policy: "StructuredToolPolicy | None" = None,
) -> tuple[Any | None, Any]:
    if isinstance(schema_source, type) and issubclass(schema_source, BaseModel):
        return await agent_obj.structured(prompt, schema_source)
    if structured_tool_policy is None:
        return await agent_obj.structured_schema(prompt, schema_source)

    from fast_agent.types import RequestParams

    return await agent_obj.structured_schema(
        prompt,
        schema_source,
        RequestParams(structured_tool_policy=structured_tool_policy),
    )


async def run_one_shot_payload(
    agent_obj: Any,
    prompt_payload: Any,
    request: "AgentRunRequest",
    structured_source: Any,
    *,
    harness_session: "HarnessSession | None" = None,
) -> "PromptMessageExtended":
    agent_name = getattr(agent_obj, "name", request.target_agent_name)
    if structured_source is None:
        response = (
            await harness_session.generate(prompt_payload, agent_name=agent_name)
            if harness_session is not None
            else await agent_obj.generate(prompt_payload)
        )
        print(response.last_text() or "")
        return response

    if harness_session is not None:
        parsed, response = await _structured_harness_call(
            harness_session,
            prompt_payload,
            structured_source,
            agent_name=agent_name,
            structured_tool_policy=request.structured_tool_policy,
        )
    else:
        parsed, response = await structured_call(
            agent_obj,
            prompt_payload,
            structured_source,
            request.structured_tool_policy,
        )
    if parsed is None:
        typer.echo(
            "Error: model response did not produce valid JSON matching the structured output schema.",
            err=True,
        )
        raise typer.Exit(1)
    sys.stdout.write(json.dumps(_structured_output_payload(parsed), ensure_ascii=False))
    return response


async def _structured_harness_call(
    harness_session: "HarnessSession",
    prompt: Any,
    schema_source: "StructuredSchemaSource",
    *,
    agent_name: str | None,
    structured_tool_policy: "StructuredToolPolicy | None" = None,
) -> tuple[Any | None, "PromptMessageExtended"]:
    if isinstance(schema_source, type) and issubclass(schema_source, BaseModel):
        return await harness_session.structured(prompt, schema_source, agent_name=agent_name)
    if structured_tool_policy is None:
        return await harness_session.structured_schema(
            prompt,
            schema_source,
            agent_name=agent_name,
        )

    from fast_agent.types import RequestParams

    return await harness_session.structured_schema(
        prompt,
        schema_source,
        agent_name=agent_name,
        request_params=RequestParams(structured_tool_policy=structured_tool_policy),
    )


def _structured_output_payload(parsed: Any) -> Any:
    if isinstance(parsed, BaseModel):
        return parsed.model_dump(mode="json")
    return parsed


__all__ = ["run_one_shot_payload", "structured_call"]
