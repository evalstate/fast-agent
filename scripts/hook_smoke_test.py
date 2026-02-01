"""Run a tool hook against a captured history for quick smoke testing."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
from pathlib import Path

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.llm_agent import LlmAgent
from fast_agent.hooks.hook_context import HookContext
from fast_agent.mcp.helpers.content_helpers import text_content
from fast_agent.mcp.prompt_serialization import load_messages, save_messages
from fast_agent.tools.hook_loader import load_hook_function
from fast_agent.types import PromptMessageExtended


@dataclass(slots=True)
class HookSmokeRunner:
    """Minimal runner for exercising tool hooks outside the full tool loop."""

    delta_messages: list[PromptMessageExtended]
    iteration: int = 0


async def _run_hook(
    *,
    hook_spec: str,
    history_path: Path,
    hook_type: str,
    output_path: Path | None,
    agent_name: str,
    base_path: Path | None,
) -> None:
    messages = load_messages(str(history_path))

    agent = LlmAgent(AgentConfig(name=agent_name))
    agent.load_message_history(messages)
    agent.set_agent_registry({agent.name: agent})

    message = (
        messages[-1]
        if messages
        else PromptMessageExtended(role="user", content=[text_content("hook smoke test")])
    )
    runner = HookSmokeRunner(delta_messages=list(messages))
    ctx = HookContext(runner=runner, agent=agent, message=message, hook_type=hook_type)

    hook_func = load_hook_function(hook_spec, base_path)

    before_count = len(agent.message_history)
    await hook_func(ctx)
    after_count = len(agent.message_history)

    print(
        "Hook executed successfully. Message history size: "
        f"{before_count} -> {after_count}."
    )

    if output_path:
        save_messages(agent.message_history, str(output_path))
        print(f"Saved updated history to {output_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hook", required=True, help="Hook spec: module.py:function")
    parser.add_argument("--history", required=True, help="History file (.json or delimited)")
    parser.add_argument(
        "--hook-type",
        default="after_turn_complete",
        help="Hook type to label in HookContext",
    )
    parser.add_argument("--output", help="Optional output path to save updated history")
    parser.add_argument("--agent-name", default="hook-smoke-test")
    parser.add_argument(
        "--base-path",
        help="Resolve relative hook paths from this directory",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    history_path = Path(args.history).expanduser()
    output_path = Path(args.output).expanduser() if args.output else None
    base_path = Path(args.base_path).expanduser() if args.base_path else None

    asyncio.run(
        _run_hook(
            hook_spec=args.hook,
            history_path=history_path,
            hook_type=args.hook_type,
            output_path=output_path,
            agent_name=args.agent_name,
            base_path=base_path,
        )
    )


if __name__ == "__main__":
    main()
