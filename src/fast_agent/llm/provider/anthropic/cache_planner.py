
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended


class AnthropicCachePlanner:
    """Calculate where to apply Anthropic cache_control blocks."""

    def __init__(
        self,
        walk_distance: int = 6,
        max_conversation_blocks: int = 2,
        max_total_blocks: int = 4,
    ) -> None:
        self.walk_distance = walk_distance
        self.max_conversation_blocks = max_conversation_blocks
        self.max_total_blocks = max_total_blocks

    def _template_prefix_count(self, messages: list[PromptMessageExtended]) -> int:
        return sum(msg.is_template for msg in messages)

    def plan_indices(
        self,
        messages: list[PromptMessageExtended],
        cache_mode: str,
        system_cache_blocks: int = 0,
    ) -> list[int]:
        """Return message indices that should receive cache_control."""

        if cache_mode == "off" or not messages:
            return []

        budget = max(0, self.max_total_blocks - system_cache_blocks)
        if budget == 0:
            return []

        template_prefix = self._template_prefix_count(messages)
        template_indices: list[int] = []

        if cache_mode in ("prompt", "auto") and template_prefix:
            template_indices = list(range(min(template_prefix, budget)))
            budget -= len(template_indices)

        conversation_indices: list[int] = []
        if cache_mode == "auto" and budget > 0:
            conv_count = max(0, len(messages) - template_prefix)
            if conv_count >= self.walk_distance:
                positions = [
                    template_prefix + i
                    for i in range(self.walk_distance - 1, conv_count, self.walk_distance)
                ]

                # Respect Anthropic limits and remaining budget
                positions = positions[-self.max_conversation_blocks :]
                conversation_indices = positions[:budget]

        return template_indices + conversation_indices
