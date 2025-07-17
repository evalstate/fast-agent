
"""
Context truncation manager for LLM conversations.
"""
import tiktoken

from mcp_agent.context import Context
from mcp_agent.context_dependent import ContextDependent
from mcp_agent.llm.memory import Memory, SimpleMemory
from mcp_agent.logging.logger import get_logger
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart

DEFAULT_SUMMARIZATION_KEEP_RATIO = 0.5 # By default, we keep 50% of the context window for recent messages when summarizing


class ContextTruncation(ContextDependent):
    """
    Manages the context window of an LLM by truncating the message history
    when it exceeds a specified token limit.

    Use truncation like this:

    @fast.agent(
        servers=[
            ... 
        ],
        use_history=True,
        request_params=RequestParams(
            maxTokens=4_096,
            max_iterations=100,

            truncation_strategy="summarize",  # Use summarization for truncation
            max_context_tokens=4_096,  # Set a maximum context token limit
        ), 
    )

    """

    def __init__(self, context: Context):
        super().__init__(context)
        self.logger = get_logger(__name__)
        self._summarization_llm = None

        self.logger.info("Initialized ContextTruncation")

    def _estimate_tokens(
        self, messages: list[PromptMessageMultipart], model: str, system_prompt: str | None = None
    ) -> int:
        """Estimate the number of tokens for a list of messages using tiktoken."""

        self.logger.info(f"_estimate_tokens(): Estimating tokens for model {model} with system prompt.")
        
        try:
            # Get the correct tokenizer for the specified model
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            # Fallback to a default tokenizer if the model is not found
            self.logger.warning(f"Model {model} not found. Using cl100k_base tokenizer.")
            encoding = tiktoken.get_encoding("cl100k_base")

        num_tokens = 0
        if system_prompt:
            # Add tokens from the system prompt
            num_tokens += len(encoding.encode(system_prompt))

        for message in messages:
            # Add tokens from each message's content
            num_tokens += len(encoding.encode(message.first_text()))
        
        # Each message adds a few extra tokens for formatting (e.g., role, content keys)
        # A common approximation is ~4 tokens per message.
        num_tokens += len(messages) * 4
        
        return num_tokens

    def needs_truncation(
        self, memory: Memory, max_tokens: int, model: str, system_prompt: str | None = None
    ) -> bool:
        """Check if the context needs to be truncated."""

        self.logger.info(f"needs_truncation() called with max_tokens: {max_tokens}.")

        if not max_tokens:
            return False
        current_tokens = self._estimate_tokens(memory.get(), model, system_prompt)
        return current_tokens > max_tokens

    def truncate(
        self, memory: Memory, max_tokens: int, model: str, system_prompt: str | None = None
    ) -> Memory:
        """
        Truncates/summarizes/compacts the memory by removing the oldest messages until the token count is within the limit.
        """

        self.logger.info(f"truncate() called with max_tokens: {max_tokens}.")

        if not self.needs_truncation(memory, max_tokens, model, system_prompt):
            return memory

        initial_tokens = self._estimate_tokens(memory.get(), model, system_prompt)
        self.logger.warning(
            f"Context ({initial_tokens} tokens) has exceeded the limit of {max_tokens} tokens. "
            "Applying simple truncation."
        )
        
        truncated_messages = list(memory.get())
        
        temp_memory = SimpleMemory()
        temp_memory.set(truncated_messages)

        while len(truncated_messages) > 1 and self.needs_truncation(
            temp_memory, max_tokens, model, system_prompt
        ):
            for i, msg in enumerate(truncated_messages):
                if msg.role != "system":
                    truncated_messages.pop(i)
                    temp_memory.set(truncated_messages)
                    break
            else:
                break
        
        final_memory = SimpleMemory()
        final_memory.set(truncated_messages)

        final_tokens = self._estimate_tokens(final_memory.get(), model, system_prompt)
        self.logger.info(
            f"Simple truncation/summarization/compaction complete. New token count: {final_tokens}"
        )

        return final_memory

    async def summarize_and_truncate(
        self, memory: Memory, max_tokens: int, model: str, system_prompt: str | None = None
    ) -> Memory:
        """
        Truncates the memory by summarizing older messages and replacing them with a summary.
        """

        self.logger.info("summarize_and_truncate() called.")

        if not self.needs_truncation(memory, max_tokens, model, system_prompt):
            return memory

        self.logger.info(f"Context has exceeded {max_tokens} tokens. Applying summarization.")

        messages = list(memory.get())
        
        system_messages = [m for m in messages if m.role == "system"]
        conversation_messages = [m for m in messages if m.role != "system"]

        split_index = self._find_summarization_point(conversation_messages, max_tokens, model)

        if split_index == 0:
            # All messages fit within the keep buffer, but the total context is still too large.
            # Fall back to simple truncation.
            return self.truncate(memory, max_tokens, model, system_prompt)

        messages_to_summarize = conversation_messages[:split_index]
        messages_to_keep = conversation_messages[split_index:]

        summary_text = await self._summarize_messages(messages_to_summarize)
        
        summary_injection = [
            PromptMessageMultipart(
                role="user", 
                content=[{"type": "text", "text": f"Here is a summary of our conversation so far: {summary_text}"}]
            ),
            PromptMessageMultipart(
                role="assistant",
                content=[{"type": "text", "text": "Thanks, I am caught up. Let's continue."}]
            )
        ]

        new_messages = system_messages + summary_injection + messages_to_keep
        
        new_memory = SimpleMemory()
        new_memory.set(new_messages)
        return new_memory

    def _find_summarization_point(
        self, messages: list[PromptMessageMultipart], max_tokens: int, model: str
    ) -> int:
        """Finds the index at which to split messages for summarization."""

        self.logger.info("Finding summarization point...")
        
        keep_buffer_tokens = int(max_tokens * DEFAULT_SUMMARIZATION_KEEP_RATIO)
        
        current_tokens = 0
        # Iterate backwards to find the messages to keep
        for i in range(len(messages) - 1, -1, -1):
            message_tokens = self._estimate_tokens([messages[i]], model)
            if current_tokens + message_tokens > keep_buffer_tokens:
                # The split point is after the current message
                return i + 1
            current_tokens += message_tokens
        
        # If all messages fit within the buffer, no summarization is needed
        return 0

# In src/mcp_agent/llm/context_truncation.py

    async def _summarize_messages(self, messages_to_summarize: list[PromptMessageMultipart]) -> str:
        """Uses an LLM to summarize a list of messages."""
        
        
        self.logger.info("!!!!!!!!!!!!!!!!!!!!!!!!!!!! _SUMMARIZE_MESSAGES !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        
        llm = self.get_summarization_llm()



        
        # Create a more concise prompt to minimize token usage
        prompt = "Summarize this conversation a maxium of five sentences:"
        messages = [PromptMessageMultipart(role="user", content=[{"type": "text", "text": prompt}])]
        messages.extend(messages_to_summarize)

        response = await llm.generate(messages)
        summary = response.first_text().strip()
        
        # Ensure the summary isn't too long
        # FIX: Use tiktoken directly instead of the missing _get_tokenizer method
        try:
            tokenizer = tiktoken.encoding_for_model("gpt-4")
        except KeyError:
            tokenizer = tiktoken.get_encoding("cl100k_base")

        if len(tokenizer.encode(summary)) > 50:  # Limit summary to ~50 tokens
            # Truncate if too long
            tokens = tokenizer.encode(summary)[:45]
            summary = tokenizer.decode(tokens) + "..."
        
        return summary



    ## TODO: Change this to always use the current LLM, not just always GPT-4.1-mini
    def get_summarization_llm(self):
        """Gets a dedicated LLM for summarization."""
        if self._summarization_llm is None:
            from mcp_agent.llm.model_factory import create_llm
            self._summarization_llm = create_llm(
                provider="openai", 
                model="gpt-4.1-mini", 
                context=self.context,
                name="summarizer"
            )
        return self._summarization_llm
