"""Task DAG — topological sort and dependency resolution for agent teams.

Provides static dependency management:
  - Kahn's topological sort for cycle detection
  - Ready-check with error policies (abort/skip/retry)
  - Cascade downstream for kill propagation
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


class CircularDependencyError(Exception):
    """Raised when a cycle is detected in the task graph."""


class DependencyError(Exception):
    """Raised when a dependency references an unknown role."""


@dataclass
class TaskNode:
    """A node in the task dependency graph."""

    role: str
    task: str = ""
    depends_on: list[str] = field(default_factory=list)
    instruction: str = ""
    servers: str = ""
    model: str = ""
    lifecycle: str = "persistent"
    on_dep_error: str = "abort"  # abort | skip | retry
    retry_max: int = 3
    metadata: dict[str, Any] = field(default_factory=dict)


class TaskDAG:
    """DAG for agent task dependencies.

    Usage::

        dag = TaskDAG.from_dict({
            "ba":  {"depends_on": [], "task": "Write specs"},
            "dev": {"depends_on": ["ba"], "task": "Implement"},
            "qe":  {"depends_on": ["dev"], "task": "Test"},
        })
        order = dag.topological_sort()  # ['ba', 'dev', 'qe']
        ready = dag.get_ready_nodes(completed={"ba"})  # [TaskNode('dev')]
    """

    def __init__(self) -> None:
        self._nodes: dict[str, TaskNode] = {}

    def add_node(self, node: TaskNode) -> None:
        self._nodes[node.role] = node

    def get_node(self, role: str) -> TaskNode | None:
        return self._nodes.get(role)

    @property
    def roles(self) -> list[str]:
        return list(self._nodes.keys())

    def validate(self) -> None:
        """Validate all dependencies exist and no cycles present."""
        for node in self._nodes.values():
            for dep in node.depends_on:
                if dep not in self._nodes:
                    raise DependencyError(
                        f"Role '{node.role}' depends on '{dep}' which is "
                        f"not defined. Available: {list(self._nodes.keys())}"
                    )
        self.topological_sort()

    def topological_sort(self) -> list[str]:
        """Kahn's algorithm. Raises CircularDependencyError on cycles."""
        in_degree: dict[str, int] = {role: 0 for role in self._nodes}
        adjacency: dict[str, list[str]] = {role: [] for role in self._nodes}

        for node in self._nodes.values():
            for dep in node.depends_on:
                if dep in adjacency:
                    adjacency[dep].append(node.role)
                    in_degree[node.role] += 1

        queue = deque(role for role, deg in in_degree.items() if deg == 0)
        result: list[str] = []

        while queue:
            role = queue.popleft()
            result.append(role)
            for dependent in adjacency.get(role, []):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        if len(result) != len(self._nodes):
            remaining = set(self._nodes.keys()) - set(result)
            raise CircularDependencyError(f"Circular dependency among: {remaining}")

        return result

    def get_ready_nodes(
        self,
        completed: set[str] | None = None,
        started: set[str] | None = None,
        failed: set[str] | None = None,
    ) -> list[TaskNode]:
        """Get nodes whose dependencies are satisfied."""
        completed = completed or set()
        started = started or set()
        failed = failed or set()

        ready: list[TaskNode] = []
        for node in self._nodes.values():
            if node.role in completed or node.role in started:
                continue

            all_deps_met = True
            for dep in node.depends_on:
                if dep in failed:
                    if node.on_dep_error == "abort":
                        all_deps_met = False
                        break
                    elif node.on_dep_error == "skip":
                        continue
                elif dep not in completed:
                    all_deps_met = False
                    break

            if all_deps_met:
                ready.append(node)

        return ready

    def get_dependents(self, role: str) -> list[str]:
        """Get roles that directly depend on the given role."""
        return [n.role for n in self._nodes.values() if role in n.depends_on]

    def get_downstream(self, role: str) -> set[str]:
        """Get all transitive dependents (for cascade/kill propagation)."""
        downstream: set[str] = set()
        queue = deque([role])
        while queue:
            current = queue.popleft()
            for dep in self.get_dependents(current):
                if dep not in downstream:
                    downstream.add(dep)
                    queue.append(dep)
        return downstream

    @classmethod
    def from_dict(cls, task_graph: dict[str, dict[str, Any]]) -> TaskDAG:
        """Build a TaskDAG from a dictionary specification."""
        dag = cls()
        for role, config in task_graph.items():
            node = TaskNode(
                role=role,
                task=config.get("task", ""),
                depends_on=config.get("depends_on", []),
                instruction=config.get("instruction", ""),
                servers=config.get("servers", ""),
                model=config.get("model", ""),
                lifecycle=config.get("lifecycle", "persistent"),
                on_dep_error=config.get("on_dep_error", "abort"),
                retry_max=config.get("retry_max", 3),
                metadata=config.get("metadata", {}),
            )
            dag.add_node(node)
        dag.validate()
        return dag

    def to_dict(self) -> dict[str, dict[str, Any]]:
        """Serialize the DAG back to a dictionary."""
        return {
            n.role: {
                "task": n.task,
                "depends_on": n.depends_on,
                "instruction": n.instruction,
                "servers": n.servers,
                "on_dep_error": n.on_dep_error,
            }
            for n in self._nodes.values()
        }
