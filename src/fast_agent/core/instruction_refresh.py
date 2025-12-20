from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Mapping
from weakref import WeakKeyDictionary


@dataclass(frozen=True)
class InstructionRefreshResult:
    updated_skill_manifests: bool = False
    updated_instruction_context: bool = False
    updated_skill_registry: bool = False
    rebuilt_instruction: bool = False


_instruction_locks: "WeakKeyDictionary[object, asyncio.Lock]" = WeakKeyDictionary()
_fallback_instruction_locks: dict[int, asyncio.Lock] = {}


def _get_instruction_lock(agent: object) -> asyncio.Lock:
    try:
        lock = _instruction_locks.get(agent)
        if lock is None:
            lock = asyncio.Lock()
            _instruction_locks[agent] = lock
        return lock
    except TypeError:
        key = id(agent)
        lock = _fallback_instruction_locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            _fallback_instruction_locks[key] = lock
        return lock


async def rebuild_agent_instruction(
    agent: object,
    *,
    skill_manifests: list[Any] | None = None,
    instruction_context: Mapping[str, str] | None = None,
    skill_registry: Any | None = None,
) -> InstructionRefreshResult:
    """Serialize rebuilds and apply optional instruction context updates."""
    lock = _get_instruction_lock(agent)
    async with lock:
        updated_skill_manifests = False
        updated_instruction_context = False
        updated_skill_registry = False
        rebuilt_instruction = False

        if skill_manifests is not None and hasattr(agent, "set_skill_manifests"):
            agent.set_skill_manifests(skill_manifests)
            updated_skill_manifests = True

        if instruction_context is not None and hasattr(agent, "set_instruction_context"):
            agent.set_instruction_context(dict(instruction_context))
            updated_instruction_context = True

        if skill_registry is not None and hasattr(agent, "skill_registry"):
            agent.skill_registry = skill_registry
            updated_skill_registry = True

        if hasattr(agent, "rebuild_instruction_templates"):
            await agent.rebuild_instruction_templates()
            rebuilt_instruction = True

        return InstructionRefreshResult(
            updated_skill_manifests=updated_skill_manifests,
            updated_instruction_context=updated_instruction_context,
            updated_skill_registry=updated_skill_registry,
            rebuilt_instruction=rebuilt_instruction,
        )
