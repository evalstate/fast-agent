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
        if clear_prompts:
            self.prompt_messages = []
