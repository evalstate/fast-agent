"""
Cache position tracking for Anthropic conversation caching.

This module provides tracking logic for Anthropic's cache control feature,
which allows caching of conversation history to reduce costs and latency.
"""

from typing import Any, List


class CachePositionTracker:
    """
    Tracks cache control positions for Anthropic conversation caching.

    Uses a walking algorithm to place cache blocks at strategic positions
    in the conversation history, limited to a maximum number of blocks.
    """

    def __init__(
        self,
        cache_walk_distance: int = 6,
        max_conversation_cache_blocks: int = 2,
    ) -> None:
        """
        Initialize cache position tracker.

        Args:
            cache_walk_distance: Number of messages between cache blocks
            max_conversation_cache_blocks: Maximum number of cache blocks to maintain
        """
        self.cache_walk_distance = cache_walk_distance
        self.max_conversation_cache_blocks = max_conversation_cache_blocks
        self.conversation_cache_positions: List[int] = []

    def should_apply_conversation_cache(self, total_messages: int) -> bool:
        """
        Determine if conversation caching should be applied.

        Args:
            total_messages: Total number of conversation messages

        Returns:
            True if we should add or update cache blocks
        """
        # Need at least cache_walk_distance messages to start caching
        if total_messages < self.cache_walk_distance:
            return False

        # Check if we need to add a new cache block
        return len(self._calculate_cache_positions(total_messages)) != len(
            self.conversation_cache_positions
        )

    def _calculate_cache_positions(self, total_conversation_messages: int) -> List[int]:
        """
        Calculate where cache blocks should be placed using walking algorithm.

        Args:
            total_conversation_messages: Number of conversation messages (not including prompts)

        Returns:
            List of positions (relative to conversation start) where cache should be placed
        """
        positions = []

        # Place cache blocks every cache_walk_distance messages
        for i in range(
            self.cache_walk_distance - 1, total_conversation_messages, self.cache_walk_distance
        ):
            positions.append(i)
            if len(positions) >= self.max_conversation_cache_blocks:
                break

        # Keep only the most recent cache blocks (walking behavior)
        if len(positions) > self.max_conversation_cache_blocks:
            positions = positions[-self.max_conversation_cache_blocks :]

        return positions

    def get_conversation_cache_updates(
        self, total_conversation_messages: int, prompt_offset: int
    ) -> dict:
        """
        Get cache position updates needed for the walking algorithm.

        Args:
            total_conversation_messages: Number of conversation messages
            prompt_offset: Offset to add to positions (for absolute positioning)

        Returns:
            Dict with 'add', 'remove', and 'active' position lists (absolute positions)
        """
        new_positions = self._calculate_cache_positions(total_conversation_messages)

        # Convert to absolute positions (including prompt messages)
        new_absolute_positions = [pos + prompt_offset for pos in new_positions]

        old_positions_set = set(self.conversation_cache_positions)
        new_positions_set = set(new_absolute_positions)

        return {
            "add": sorted(new_positions_set - old_positions_set),
            "remove": sorted(old_positions_set - new_positions_set),
            "active": sorted(new_absolute_positions),
        }

    def apply_conversation_cache_updates(self, updates: dict) -> None:
        """
        Apply cache position updates.

        Args:
            updates: Dict from get_conversation_cache_updates()
        """
        self.conversation_cache_positions = updates["active"].copy()

    def clear(self) -> None:
        """Clear all tracked cache positions."""
        self.conversation_cache_positions = []

    @staticmethod
    def remove_cache_control_from_messages(
        messages: List[Any], positions: List[int]
    ) -> None:
        """
        Remove cache control from specified message positions.

        Args:
            messages: The message array to modify
            positions: List of positions to remove cache control from
        """
        for pos in positions:
            if pos < len(messages):
                message = messages[pos]
                if isinstance(message, dict) and "content" in message:
                    content_list = message["content"]
                    if isinstance(content_list, list):
                        for content_block in content_list:
                            if isinstance(content_block, dict) and "cache_control" in content_block:
                                del content_block["cache_control"]

    @staticmethod
    def add_cache_control_to_messages(messages: List[Any], positions: List[int]) -> int:
        """
        Add cache control to specified message positions.

        Args:
            messages: The message array to modify
            positions: List of positions to add cache control to

        Returns:
            Number of cache blocks successfully applied
        """
        applied_count = 0
        for pos in positions:
            if pos < len(messages):
                message = messages[pos]
                if isinstance(message, dict) and "content" in message:
                    content_list = message["content"]
                    if isinstance(content_list, list) and content_list:
                        # Apply cache control to the last content block
                        for content_block in reversed(content_list):
                            if isinstance(content_block, dict):
                                content_block["cache_control"] = {"type": "ephemeral"}
                                applied_count += 1
                                break
        return applied_count
