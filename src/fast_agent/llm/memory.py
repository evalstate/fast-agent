from typing import Generic, Protocol, TypeVar

# Define our own type variable for implementation use
MessageParamT = TypeVar("MessageParamT")


class Memory(Protocol, Generic[MessageParamT]):
    """
    Simple memory management for storing past interactions in-memory.

    IMPORTANT: As of the conversation history architecture refactor,
    provider history is DIAGNOSTIC ONLY. Messages are generated fresh
    from _message_history on each API call via _convert_to_provider_format().

    The get() method should NOT be called by provider code for API calls.
    It may still be used for debugging/inspection purposes.
    """

    # TODO: saqadri - add checkpointing and other advanced memory capabilities

    def __init__(self) -> None: ...

    def extend(self, messages: list[MessageParamT], is_prompt: bool = False) -> None: ...

    def set(self, messages: list[MessageParamT], is_prompt: bool = False) -> None: ...

    def append(self, message: MessageParamT, is_prompt: bool = False) -> None: ...

    def get(self, include_completion_history: bool = True) -> list[MessageParamT]: ...

    def clear(self, clear_prompts: bool = False) -> None: ...

    def pop(self, *, from_prompts: bool = False) -> MessageParamT | None: ...


class SimpleMemory(Memory, Generic[MessageParamT]):
    """
    Simple memory management for storing past interactions in-memory.

    Maintains both prompt messages (which are always included) and
    generated conversation history (which is included based on use_history setting).
    """

    def __init__(self) -> None:
        self.history: list[MessageParamT] = []
        self.prompt_messages: list[MessageParamT] = []  # Always included

    def extend(self, messages: list[MessageParamT], is_prompt: bool = False) -> None:
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

    def set(self, messages: list[MessageParamT], is_prompt: bool = False) -> None:
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

    def get(self, include_completion_history: bool = True) -> list[MessageParamT]:
        """
        Get all messages in memory.

        DEPRECATED: Provider history is now diagnostic only. This method returns
        a diagnostic snapshot and should NOT be used for API calls. Messages for
        API calls are generated fresh from _message_history via
        _convert_to_provider_format().

        Args:
            include_history: If True, include regular history messages
                             If False, only return prompt messages

        Returns:
            Combined list of prompt messages and optionally history messages
            (for diagnostic/inspection purposes only)
        """
        # Note: We don't emit a warning here because this method is still
        # legitimately used for diagnostic purposes and by some internal code.
        # The important change is that provider completion methods no longer
        # call this for API message construction.
        if include_completion_history:
            return self.prompt_messages + self.history
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

    def pop(self, *, from_prompts: bool = False) -> MessageParamT | None:
        """
        Remove and return the most recent message from history or prompt messages.

        Args:
            from_prompts: If True, pop from prompt_messages instead of history

        Returns:
            The removed message if available, otherwise None
        """
        if from_prompts:
            if not self.prompt_messages:
                return None
            return self.prompt_messages.pop()

        if not self.history:
            return None

        return self.history.pop()
