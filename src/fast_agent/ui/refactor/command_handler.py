"""
Command handler skeleton: delegates to existing InteractivePrompt for now.
This allows ACP and TUI to import a CommandHandler while we incrementally move logic.
"""

from typing import Any

from fast_agent.ui.interactive_prompt import InteractivePrompt


class CommandHandler:
    """Thin wrapper around InteractivePrompt to centralize command entry point.
    Later this will contain command implementations moved out of the prompt loop.
    """

    def __init__(self, agent_types: dict[str, Any] | None = None) -> None:
        self._impl = InteractivePrompt(agent_types=agent_types)

    # Keep compatible signature for incremental migration
    async def handle(self, payload, agent, prompt_provider, display):
        # For now call into InteractivePrompt methods as appropriate.
        # TODO: move command implementations here.
        raise NotImplementedError("CommandHandler.handle should be implemented during refactor")


__all__ = ["CommandHandler"]
