"""Digests of full tool surfaces, pinned in ``{direct,profile}_surface_v0.json``.

``test_full_surface_matches_parent`` compares each row's digest with the current surface. The
rows were captured from git archive 146a72df8; when a change deliberately alters a surface,
refresh the digests and review exactly what moved:

    uv run python -m tests.unit.fast_agent.ptc.surface_fixtures          # report changed rows
    uv run python -m tests.unit.fast_agent.ptc.surface_fixtures --write  # rewrite their digests

Only the ``sha256`` values are replaced, so the files keep their layout.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypedDict

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.mcp_agent import McpAgent
from fast_agent.agents.subagent_tool import install_subagent_tool
from fast_agent.config import ProgrammaticToolCallingSettings, Settings, ShellSettings
from fast_agent.context import Context
from fast_agent.core.harness_tools import set_harness_tools
from fast_agent.ptc.vocabulary import PTC_VOCABULARY_META
from fast_agent.tools.tool_surface import ToolExposure

if TYPE_CHECKING:
    from collections.abc import Sequence

    from fast_agent.tools.shell_profiles import ShellToolProfile

type SurfaceMode = Literal["direct", "profile"]
SURFACE_MODES: tuple[SurfaceMode, ...] = ("direct", "profile")


class SurfaceRow(TypedDict):
    model: str
    profile: ShellToolProfile
    extras: bool
    sha256: str


def surface_path(mode: SurfaceMode) -> Path:
    return Path(__file__).with_name(f"{mode}_surface_v0.json")


def surface_rows(mode: SurfaceMode) -> list[SurfaceRow]:
    return json.loads(surface_path(mode).read_text())["rows"]


async def surface_record(mode: SurfaceMode, row: SurfaceRow) -> dict[str, Any]:
    """Full definitions, plus PTC prose, function order, signatures and stubs in profile mode.

    Only the additive vocabulary marker is excluded; nothing else is normalized.
    """
    enabled = mode == "profile"
    agent = McpAgent(
        AgentConfig(
            name="t",
            instruction="x",
            servers=[],
            shell=True,
            model=row["model"],
            subagents=row["extras"],
        ),
        context=Context(
            config=Settings(
                shell_execution=ShellSettings(
                    tool_profile=row["profile"], enable_attach_media="on"
                ),
                ptc=ProgrammaticToolCallingSettings(function_names="profile"),
            ),
            ptc_enabled=enabled,
        ),
    )
    try:
        if row["extras"]:
            install_subagent_tool(agent)
            set_harness_tools(agent, True)
        tools = [
            tool.model_dump(mode="json", by_alias=True, exclude_none=True)
            for tool in (await agent.list_tools()).tools
        ]
        for tool in tools:
            tool.get("_meta", {}).pop(PTC_VOCABULARY_META, None)
        record: dict[str, Any] = {"tools": tools}
        if enabled:
            surface = await agent.tool_surface()
            assert surface.programmatic is not None
            programmatic = asdict(surface.programmatic)
            programmatic.pop("vocabulary")
            record["programmatic"] = programmatic
            record["functions"] = [
                (entry.python_name, entry.tool.name, entry.signature, entry.stub)
                for entry in surface.exposed(ToolExposure.PROGRAMMATIC)
            ]
        return record
    finally:
        await agent.shutdown()


def surface_digest(record: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


_DIGEST_FIELD = re.compile(r'("sha256": ")[0-9a-f]{64}(")')


def replace_digests(text: str, digests: Sequence[str]) -> str:
    """Replace each row's digest in file order, leaving the layout untouched."""
    pending = iter(digests)
    result, count = _DIGEST_FIELD.subn(lambda m: f"{m[1]}{next(pending)}{m[2]}", text)
    if count != len(digests):
        raise ValueError(f"{count} digest fields for {len(digests)} rows")
    return result


async def refresh(*, write: bool) -> None:
    for mode in SURFACE_MODES:
        path = surface_path(mode)
        rows = surface_rows(mode)
        digests = [surface_digest(await surface_record(mode, row)) for row in rows]
        changed = sum(digest != row["sha256"] for digest, row in zip(digests, rows, strict=True))
        print(f"{mode}: {changed} of {len(rows)} rows changed")
        if write:
            path.write_text(replace_digests(path.read_text(), digests))


if __name__ == "__main__":
    asyncio.run(refresh(write="--write" in sys.argv))
