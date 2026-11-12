"""Import-boundary contracts for workflow decorator defaults."""

import subprocess
import sys


def test_direct_decorators_does_not_import_workflow_implementations() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys

import fast_agent.core.direct_decorators

implementations = {
    "fast_agent.agents.workflow.iterative_planner",
    "fast_agent.agents.workflow.router_agent",
}
loaded = implementations.intersection(sys.modules)
assert not loaded, f"Decorator import loaded workflow implementations: {sorted(loaded)}"
""",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_workflow_prompt_exports_are_preserved() -> None:
    from fast_agent.agents.workflow import prompts
    from fast_agent.agents.workflow.iterative_planner import ITERATIVE_PLAN_SYSTEM_PROMPT_TEMPLATE
    from fast_agent.agents.workflow.router_agent import ROUTING_SYSTEM_INSTRUCTION

    assert ITERATIVE_PLAN_SYSTEM_PROMPT_TEMPLATE is prompts.ITERATIVE_PLAN_SYSTEM_PROMPT_TEMPLATE
    assert ROUTING_SYSTEM_INSTRUCTION is prompts.ROUTING_SYSTEM_INSTRUCTION
