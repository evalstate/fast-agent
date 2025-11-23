"""
Keep track of all activities/tasks that the executor needs to run.
This is used by the workflow engine to dynamically orchestrate a workflow graph.
The user just writes standard functions annotated with @workflow_task, but behind the scenes a workflow graph is built.
"""

from typing import Any, Callable


class ActivityRegistry:
    """Centralized task/activity management with validation and metadata."""

    def __init__(self) -> None:
        self._activities: dict[str, Callable] = {}
        self._metadata: dict[str, dict[str, Any]] = {}

    def register(self, name: str, func: Callable, metadata: dict[str, Any] | None = None) -> None:
        if name in self._activities:
            raise ValueError(f"Activity '{name}' is already registered.")
        self._activities[name] = func
        self._metadata[name] = metadata or {}

    def get_activity(self, name: str) -> Callable:
        if name not in self._activities:
            raise KeyError(f"Activity '{name}' not found.")
        return self._activities[name]

    def get_metadata(self, name: str) -> dict[str, Any]:
        return self._metadata.get(name, {})

    def list_activities(self) -> list[str]:
        return list(self._activities.keys())
