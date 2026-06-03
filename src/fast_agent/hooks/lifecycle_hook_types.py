from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

LifecycleHookType = Literal["on_start", "on_shutdown"]
LifecycleHookProgressKind = Literal["agent_startup", "agent_shutdown"]


@dataclass(frozen=True, slots=True)
class LifecycleHookDescriptor:
    hook_type: LifecycleHookType
    attr_name: LifecycleHookType
    progress_kind: LifecycleHookProgressKind
    raises_on_failure: bool


LIFECYCLE_HOOK_DESCRIPTORS: dict[LifecycleHookType, LifecycleHookDescriptor] = {
    "on_start": LifecycleHookDescriptor(
        hook_type="on_start",
        attr_name="on_start",
        progress_kind="agent_startup",
        raises_on_failure=True,
    ),
    "on_shutdown": LifecycleHookDescriptor(
        hook_type="on_shutdown",
        attr_name="on_shutdown",
        progress_kind="agent_shutdown",
        raises_on_failure=False,
    ),
}
VALID_LIFECYCLE_HOOK_TYPES = frozenset(LIFECYCLE_HOOK_DESCRIPTORS)


def lifecycle_hook_descriptor(hook_type: LifecycleHookType) -> LifecycleHookDescriptor:
    return LIFECYCLE_HOOK_DESCRIPTORS[hook_type]
