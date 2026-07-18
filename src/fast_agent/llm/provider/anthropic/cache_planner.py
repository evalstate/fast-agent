from fast_agent.mcp.prompt_message_extended import PromptMessageExtended


class AnthropicCachePlanner:
    """Calculate where to apply Anthropic cache_control blocks."""

    def __init__(
        self,
        max_conversation_blocks: int = 2,
        max_total_blocks: int = 4,
    ) -> None:
        self.max_conversation_blocks = max_conversation_blocks
        self.max_total_blocks = max_total_blocks

    def _template_prefix_count(self, messages: list[PromptMessageExtended]) -> int:
        return sum(msg.is_template for msg in messages)

    def plan_indices(
        self,
        messages: list[PromptMessageExtended],
        cache_mode: str,
        system_cache_blocks: int = 0,
        process_poll_boundary: int | None = None,
    ) -> list[int]:
        """Return message indices that should receive cache_control."""

        if cache_mode == "off" or not messages:
            return []

        budget = max(0, self.max_total_blocks - system_cache_blocks)
        if budget == 0:
            return []

        template_prefix = self._template_prefix_count(messages)
        template_indices: list[int] = []

        process_indices: list[int] = []
        if (
            cache_mode == "auto"
            and budget > 0
            and process_poll_boundary is not None
            and 0 <= process_poll_boundary < len(messages)
        ):
            process_indices = [process_poll_boundary]
            budget -= 1

        conversation_candidates = [
            index
            for index in range(template_prefix, len(messages))
            if messages[index].role == "assistant"
        ]
        conversation_reserve = int(
            cache_mode == "auto" and not process_indices and bool(conversation_candidates)
        )

        if cache_mode in ("prompt", "auto") and template_prefix:
            template_indices = [
                index
                for index in range(template_prefix)
                if index != process_poll_boundary
            ][: max(0, budget - conversation_reserve)]
            budget -= len(template_indices)

        conversation_indices: list[int] = []
        if cache_mode == "auto" and budget > 0 and not process_indices:
            # Reapply the prior checkpoint and advance it to the newest assistant turn.
            conversation_indices = conversation_candidates[
                -min(budget, self.max_conversation_blocks) :
            ]

        return sorted(template_indices + process_indices + conversation_indices)
