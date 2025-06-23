from typing import Generic, List, Protocol, TypeVar

# Define our own type variable for implementation use
MessageParamT = TypeVar("MessageParamT")


class Memory(Protocol, Generic[MessageParamT]):
    """
    Simple memory management for storing past interactions in-memory.
    """

    # TODO: saqadri - add checkpointing and other advanced memory capabilities

    def __init__(self) -> None: ...

    def extend(self, messages: List[MessageParamT], is_prompt: bool = False) -> None: ...

    def set(self, messages: List[MessageParamT], is_prompt: bool = False) -> None: ...

    def append(self, message: MessageParamT, is_prompt: bool = False) -> None: ...

    def get(self, include_completion_history: bool = True) -> List[MessageParamT]: ...

    def clear(self, clear_prompts: bool = False) -> None: ...


class SimpleMemory(Memory, Generic[MessageParamT]):
    """
    Simple memory management for storing past interactions in-memory.

    Maintains both prompt messages (which are always included) and
    generated conversation history (which is included based on use_history setting).
    """

    def __init__(self) -> None:
        self.history: List[MessageParamT] = []
        self.prompt_messages: List[MessageParamT] = []  # Always included
        self.conversation_cache_positions: List[int] = []  # Track active conversation cache positions
        self.cache_walk_distance: int = 6  # Messages between cache blocks
        self.max_conversation_cache_blocks: int = 2  # Maximum conversation cache blocks

    def extend(self, messages: List[MessageParamT], is_prompt: bool = False) -> None:
        """
        Add multiple messages to history.

        Args:
            messages: Messages to add
            is_prompt: If True, add to prompt_messages instead of regular history
        """
        if is_prompt:
            self.prompt_messages.extend(messages)
        else:
            self.history.extend(messages)

    def set(self, messages: List[MessageParamT], is_prompt: bool = False) -> None:
        """
        Replace messages in history.

        Args:
            messages: Messages to set
            is_prompt: If True, replace prompt_messages instead of regular history
        """
        if is_prompt:
            self.prompt_messages = messages.copy()
        else:
            self.history = messages.copy()

    def append(self, message: MessageParamT, is_prompt: bool = False) -> None:
        """
        Add a single message to history.

        Args:
            message: Message to add
            is_prompt: If True, add to prompt_messages instead of regular history
        """
        if is_prompt:
            self.prompt_messages.append(message)
        else:
            self.history.append(message)

    def get(self, include_completion_history: bool = True) -> List[MessageParamT]:
        """
        Get all messages in memory.

        Args:
            include_history: If True, include regular history messages
                             If False, only return prompt messages

        Returns:
            Combined list of prompt messages and optionally history messages
        """
        if include_completion_history:
            return self.prompt_messages + self.history
        else:
            return self.prompt_messages.copy()

    def clear(self, clear_prompts: bool = False) -> None:
        """
        Clear history and optionally prompt messages.

        Args:
            clear_prompts: If True, also clear prompt messages
        """
        self.history = []
        self.conversation_cache_positions = []  # Reset cache positions
        if clear_prompts:
            self.prompt_messages = []

    def should_apply_conversation_cache(self) -> bool:
        """
        Determine if conversation caching should be applied based on walking algorithm.
        
        Returns:
            True if we should add or update cache blocks
        """
        total_messages = len(self.history)
        
        # Need at least cache_walk_distance messages to start caching
        if total_messages < self.cache_walk_distance:
            return False
            
        # Check if we need to add a new cache block
        return len(self._calculate_cache_positions(total_messages)) != len(self.conversation_cache_positions)
    
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
        for i in range(self.cache_walk_distance - 1, total_conversation_messages, self.cache_walk_distance):
            positions.append(i)
            if len(positions) >= self.max_conversation_cache_blocks:
                break
        
        # Keep only the most recent cache blocks (walking behavior)
        if len(positions) > self.max_conversation_cache_blocks:
            positions = positions[-self.max_conversation_cache_blocks:]
            
        return positions
    
    def get_conversation_cache_updates(self) -> dict:
        """
        Get cache position updates needed for the walking algorithm.
        
        Returns:
            Dict with 'add', 'remove', and 'active' position lists (relative to full message array)
        """
        total_conversation_messages = len(self.history)
        new_positions = self._calculate_cache_positions(total_conversation_messages)
        
        # Convert to absolute positions (including prompt messages)
        prompt_offset = len(self.prompt_messages)
        new_absolute_positions = [pos + prompt_offset for pos in new_positions]
        
        old_positions_set = set(self.conversation_cache_positions)
        new_positions_set = set(new_absolute_positions)
        
        return {
            'add': sorted(new_positions_set - old_positions_set),
            'remove': sorted(old_positions_set - new_positions_set), 
            'active': sorted(new_absolute_positions)
        }
    
    def apply_conversation_cache_updates(self, updates: dict) -> None:
        """
        Apply cache position updates.
        
        Args:
            updates: Dict from get_conversation_cache_updates()
        """
        self.conversation_cache_positions = updates['active'].copy()

    def remove_cache_control_from_messages(self, messages: List[MessageParamT], positions: List[int]) -> None:
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

    def add_cache_control_to_messages(self, messages: List[MessageParamT], positions: List[int]) -> int:
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
