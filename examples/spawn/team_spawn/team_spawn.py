"""Team Spawn — example orchestrating a multi-agent team.

Demonstrates spawning a full agile team from a YAML template
with dependency-resolved execution order.
"""

import asyncio

from fast_agent.spawn.spawn_registry import SpawnRegistry
from fast_agent.spawn.team_spawner import spawn_team


async def main() -> None:
    registry = SpawnRegistry(registry_file=".runtime/state/spawn_registry.json")

    session = await spawn_team(
        template_name="agile_team",
        project_brief="Build a CLI calculator that supports +, -, *, /",
        registry=registry,
        project_dir=".",
        template_dir="team_templates",
    )

    print(f"Session ID: {session.session_id}")
    print(f"Sprint status: {session.sprint_status}")
    for step_name, info in session.step_runs.items():
        print(f"  {step_name}: {info['status']}")


if __name__ == "__main__":
    asyncio.run(main())
