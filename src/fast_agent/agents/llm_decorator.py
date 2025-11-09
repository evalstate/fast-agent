"""
Decorator for LlmAgent, normalizes PromptMessageExtended, allows easy extension of Agents
"""

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)

if TYPE_CHECKING:
    from rich.text import Text

from a2a.types import AgentCard
from mcp import ListToolsResult, Tool
from mcp.types import (
    CallToolResult,
    ContentBlock,
    EmbeddedResource,
    GetPromptResult,
    ImageContent,
    Prompt,
    PromptMessage,
    ReadResourceResult,
    ResourceLink,
    TextContent,
    TextResourceContents,
)
from opentelemetry import trace
from pydantic import BaseModel

from fast_agent.agents.agent_types import AgentConfig, AgentType
from fast_agent.constants import FAST_AGENT_ERROR_CHANNEL, FAST_AGENT_REMOVED_METADATA_CHANNEL
from fast_agent.context import Context
from fast_agent.interfaces import (
    AgentProtocol,
    FastAgentLLMProtocol,
    LLMFactoryProtocol,
    StreamingAgentProtocol,
)
from fast_agent.llm.model_database import ModelDatabase
from fast_agent.llm.provider_types import Provider
from fast_agent.llm.usage_tracking import UsageAccumulator
from fast_agent.mcp.helpers.content_helpers import normalize_to_extended_list, text_content
from fast_agent.mcp.mime_utils import is_text_mime_type
from fast_agent.types import PromptMessageExtended, RequestParams

# Define a TypeVar for models
ModelT = TypeVar("ModelT", bound=BaseModel)

LLM = TypeVar("LLM", bound=FastAgentLLMProtocol)


class StreamingNotAvailableError(RuntimeError):
    """Raised when streaming APIs are accessed before an LLM is attached."""


class StreamingAgentMixin(StreamingAgentProtocol):
    """Mixin that forwards streaming listener registration to the attached LLM."""

    def add_stream_listener(self, listener: Callable[[str], None]) -> Callable[[], None]:
        try:
            llm = self.llm
        except AssertionError as exc:  # LLM not attached yet
            raise StreamingNotAvailableError(
                "LLM not attached; cannot register stream listener"
            ) from exc
        return llm.add_stream_listener(listener)

    def add_tool_stream_listener(
        self, listener: Callable[[str, Dict[str, Any] | None], None]
    ) -> Callable[[], None]:
        try:
            llm = self.llm
        except AssertionError as exc:
            raise StreamingNotAvailableError(
                "LLM not attached; cannot register tool stream listener"
            ) from exc
        return llm.add_tool_stream_listener(listener)


@dataclass
class _RemovedBlock:
    """Internal representation of a removed content block."""

    category: str
    mime_type: str | None
    source: str
    tool_id: str | None
    block: ContentBlock


@dataclass(frozen=True)
class RemovedContentSummary:
    """Summary information about removed content for the last turn."""

    model_name: str | None
    counts: Dict[str, int]
    category_mimes: Dict[str, Tuple[str, ...]]
    alert_flags: frozenset[str]
    message: str


class LlmDecorator(StreamingAgentMixin, AgentProtocol):
    """
    A pure delegation wrapper around LlmAgent instances.

    This class provides simple delegation to an attached LLM without adding
    any LLM interaction behaviors. Subclasses can add specialized logic
    for stop reason handling, UI display, tool execution, etc.

    Stub implementations of advanced convenience methods are supplied.
    """

    def __init__(
        self,
        config: AgentConfig,
        context: Context | None = None,
    ) -> None:
        self.config = config

        self._context = context
        self._name = self.config.name
        self._tracer = trace.get_tracer(__name__)
        self.instruction = self.config.instruction

        # Store the default request params from config
        self._default_request_params = self.config.default_request_params

        # Initialize the LLM to None (will be set by attach_llm)
        self._llm: Optional[FastAgentLLMProtocol] = None
        self._initialized = False

    @property
    def context(self) -> Context | None:
        """Optional execution context supplied at construction time."""
        return self._context

    @property
    def initialized(self) -> bool:
        """Check if the agent is initialized."""
        return self._initialized

    @initialized.setter
    def initialized(self, value: bool) -> None:
        """Set the initialized state."""
        self._initialized = value

    async def initialize(self) -> None:
        self.initialized = True

    async def shutdown(self) -> None:
        self.initialized = False

    @property
    def agent_type(self) -> AgentType:
        """
        Return the type of this agent.
        """
        return AgentType.LLM

    @property
    def name(self) -> str:
        """
        Return the name of this agent.
        """
        return self._name

    async def attach_llm(
        self,
        llm_factory: LLMFactoryProtocol,
        model: str | None = None,
        request_params: RequestParams | None = None,
        **additional_kwargs,
    ) -> FastAgentLLMProtocol:
        """
        Create and attach an LLM instance to this agent.

        Parameters have the following precedence (highest to lowest):
        1. Explicitly passed parameters to this method
        2. Agent's default_request_params
        3. LLM's default values

        Args:
            llm_factory: A factory function that constructs an AugmentedLLM
            model: Optional model name override
            request_params: Optional request parameters override
            **additional_kwargs: Additional parameters passed to the LLM constructor

        Returns:
            The created LLM instance
        """
        # Merge parameters with proper precedence
        effective_params = self._merge_request_params(
            self._default_request_params, request_params, model
        )

        # Create the LLM instance
        self._llm = llm_factory(
            agent=self, request_params=effective_params, context=self._context, **additional_kwargs
        )

        return self._llm

    async def __call__(
        self,
        message: Union[
            str,
            PromptMessage,
            PromptMessageExtended,
            Sequence[Union[str, PromptMessage, PromptMessageExtended]],
        ],
    ) -> str:
        """
        Make the agent callable to send messages.

        Args:
            message: Optional message to send to the agent

        Returns:
            The agent's response as a string
        """
        return await self.send(message)

    async def send(
        self,
        message: Union[
            str,
            PromptMessage,
            PromptMessageExtended,
            Sequence[Union[str, PromptMessage, PromptMessageExtended]],
        ],
        request_params: RequestParams | None = None,
    ) -> str:
        """
        Convenience method to generate and return a string directly
        """
        response = await self.generate(message, request_params)
        return response.last_text() or ""

    async def generate(
        self,
        messages: Union[
            str,
            PromptMessage,
            PromptMessageExtended,
            Sequence[Union[str, PromptMessage, PromptMessageExtended]],
        ],
        request_params: RequestParams | None = None,
        tools: List[Tool] | None = None,
    ) -> PromptMessageExtended:
        """
        Create a completion with the LLM using the provided messages.

        This method provides the friendly agent interface by normalizing inputs
        and delegating to generate_impl.

        Args:
            messages: Message(s) in various formats:
                - String: Converted to a user PromptMessageExtended
                - PromptMessage: Converted to PromptMessageExtended
                - PromptMessageExtended: Used directly
                - List of any combination of the above
            request_params: Optional parameters to configure the request
            tools: Optional list of tools available to the LLM

        Returns:
            The LLM's response as a PromptMessageExtended
        """
        # Normalize all input types to a list of PromptMessageExtended
        multipart_messages = normalize_to_extended_list(messages)
        final_request_params = (
            self.llm.get_request_params(request_params) if self._llm else request_params
        )

        with self._tracer.start_as_current_span(f"Agent: '{self._name}' generate"):
            return await self.generate_impl(multipart_messages, final_request_params, tools)

    async def generate_impl(
        self,
        messages: List[PromptMessageExtended],
        request_params: RequestParams | None = None,
        tools: List[Tool] | None = None,
    ) -> PromptMessageExtended:
        """
        Implementation method for generate.

        Default implementation delegates to the attached LLM.
        Subclasses can override this to customize behavior while still
        benefiting from the message normalization in generate().

        Args:
            messages: Normalized list of PromptMessageExtended objects
            request_params: Optional parameters to configure the request
            tools: Optional list of tools available to the LLM

        Returns:
            The LLM's response as a PromptMessageExtended
        """
        response, _ = await self._generate_with_summary(messages, request_params, tools)
        return response

    async def apply_prompt_template(self, prompt_result: GetPromptResult, prompt_name: str) -> str:
        """
        Apply a prompt template as persistent context that will be included in all future conversations.
        Delegates to the attached LLM.

        Args:
            prompt_result: The GetPromptResult containing prompt messages
            prompt_name: The name of the prompt being applied

        Returns:
            String representation of the assistant's response if generated
        """
        assert self._llm
        return await self._llm.apply_prompt_template(prompt_result, prompt_name)

    async def apply_prompt(
        self,
        prompt: Union[str, GetPromptResult],
        arguments: Dict[str, str] | None = None,
        as_template: bool = False,
        namespace: str | None = None,
    ) -> str:
        """
        Default, provider-agnostic apply_prompt implementation.

        - If given a GetPromptResult, optionally store as template or generate once.
        - If given a string, treat it as plain user text and generate.

        Subclasses that integrate MCP servers should override this.
        """
        # If a prompt template object is provided
        if isinstance(prompt, GetPromptResult):
            namespaced_name = getattr(prompt, "namespaced_name", "template")
            if as_template:
                return await self.apply_prompt_template(prompt, namespaced_name)

            messages = PromptMessageExtended.from_get_prompt_result(prompt)
            response = await self.generate_impl(messages, None)
            return response.first_text()

        # Otherwise treat the string as plain content (ignore arguments here)
        return await self.send(prompt)

    def clear(self, *, clear_prompts: bool = False) -> None:
        """Reset conversation state while optionally retaining applied prompt templates."""

        if not self._llm:
            return
        self._llm.clear(clear_prompts=clear_prompts)

    async def structured(
        self,
        messages: Union[
            str,
            PromptMessage,
            PromptMessageExtended,
            Sequence[Union[str, PromptMessage, PromptMessageExtended]],
        ],
        model: Type[ModelT],
        request_params: RequestParams | None = None,
    ) -> Tuple[ModelT | None, PromptMessageExtended]:
        """
        Apply the prompt and return the result as a Pydantic model.

        This method provides the friendly agent interface by normalizing inputs
        and delegating to structured_impl.

        Args:
            messages: Message(s) in various formats:
                - String: Converted to a user PromptMessageExtended
                - PromptMessage: Converted to PromptMessageExtended
                - PromptMessageExtended: Used directly
                - List of any combination of the above
            model: The Pydantic model class to parse the result into
            request_params: Optional parameters to configure the LLM request

        Returns:
            A tuple of (parsed model instance or None, assistant response message)
        """
        # Normalize all input types to a list of PromptMessageExtended
        multipart_messages = normalize_to_extended_list(messages)
        final_request_params = (
            self.llm.get_request_params(request_params) if self._llm else request_params
        )

        with self._tracer.start_as_current_span(f"Agent: '{self._name}' structured"):
            return await self.structured_impl(multipart_messages, model, final_request_params)

    async def structured_impl(
        self,
        messages: List[PromptMessageExtended],
        model: Type[ModelT],
        request_params: RequestParams | None = None,
    ) -> Tuple[ModelT | None, PromptMessageExtended]:
        """
        Implementation method for structured.

        Default implementation delegates to the attached LLM.
        Subclasses can override this to customize behavior while still
        benefiting from the message normalization in structured().

        Args:
            messages: Normalized list of PromptMessageExtended objects
            model: The Pydantic model class to parse the result into
            request_params: Optional parameters to configure the LLM request

        Returns:
            A tuple of (parsed model instance or None, assistant response message)
        """
        result, _ = await self._structured_with_summary(messages, model, request_params)
        return result

    async def _generate_with_summary(
        self,
        messages: List[PromptMessageExtended],
        request_params: RequestParams | None = None,
        tools: List[Tool] | None = None,
    ) -> Tuple[PromptMessageExtended, RemovedContentSummary | None]:
        assert self._llm, "LLM is not attached"
        sanitized_messages, summary = self._sanitize_messages_for_llm(messages)
        response = await self._llm.generate(sanitized_messages, request_params, tools)
        return response, summary

    async def _structured_with_summary(
        self,
        messages: List[PromptMessageExtended],
        model: Type[ModelT],
        request_params: RequestParams | None = None,
    ) -> Tuple[Tuple[ModelT | None, PromptMessageExtended], RemovedContentSummary | None]:
        assert self._llm, "LLM is not attached"
        sanitized_messages, summary = self._sanitize_messages_for_llm(messages)
        structured_result = await self._llm.structured(sanitized_messages, model, request_params)
        return structured_result, summary

    def _sanitize_messages_for_llm(
        self, messages: List[PromptMessageExtended]
    ) -> Tuple[List[PromptMessageExtended], RemovedContentSummary | None]:
        """Filter out content blocks that the current model cannot tokenize."""
        if not messages:
            return [], None

        removed_blocks: List[_RemovedBlock] = []
        sanitized_messages: List[PromptMessageExtended] = []

        for message in messages:
            sanitized, removed = self._sanitize_message_for_llm(message)
            sanitized_messages.append(sanitized)
            removed_blocks.extend(removed)

        summary = self._build_removed_summary(removed_blocks)
        if summary:
            # Attach metadata to the last user message for downstream UI usage
            for msg in reversed(sanitized_messages):
                if msg.role == "user":
                    channels = dict(msg.channels or {})
                    meta_entries = list(channels.get(FAST_AGENT_REMOVED_METADATA_CHANNEL, []))
                    meta_entries.extend(self._build_metadata_entries(removed_blocks))
                    channels[FAST_AGENT_REMOVED_METADATA_CHANNEL] = meta_entries
                    msg.channels = channels
                    break

        return sanitized_messages, summary

    def _sanitize_message_for_llm(
        self, message: PromptMessageExtended
    ) -> Tuple[PromptMessageExtended, List[_RemovedBlock]]:
        """Return a sanitized copy of a message and any removed content blocks."""
        msg_copy = message.model_copy(deep=True)
        removed: List[_RemovedBlock] = []

        msg_copy.content = self._filter_block_list(
            list(msg_copy.content or []), removed, source="message"
        )

        if msg_copy.tool_results:
            new_tool_results: Dict[str, CallToolResult] = {}
            for tool_id, tool_result in msg_copy.tool_results.items():
                original_blocks = list(tool_result.content or [])
                filtered_blocks = self._filter_block_list(
                    original_blocks,
                    removed,
                    source="tool_result",
                    tool_id=tool_id,
                )

                if filtered_blocks != original_blocks:
                    try:
                        updated_result = tool_result.model_copy(update={"content": filtered_blocks})
                    except AttributeError:
                        updated_result = CallToolResult(
                            content=filtered_blocks, isError=getattr(tool_result, "isError", False)
                        )
                else:
                    updated_result = tool_result

                new_tool_results[tool_id] = updated_result

            msg_copy.tool_results = new_tool_results

        if removed:
            channels = dict(msg_copy.channels or {})
            error_entries = list(channels.get(FAST_AGENT_ERROR_CHANNEL, []))
            error_entries.extend(self._build_error_channel_entries(removed))
            channels[FAST_AGENT_ERROR_CHANNEL] = error_entries
            msg_copy.channels = channels

        return msg_copy, removed

    def _filter_block_list(
        self,
        blocks: Sequence[ContentBlock],
        removed: List[_RemovedBlock],
        *,
        source: str,
        tool_id: str | None = None,
    ) -> List[ContentBlock]:
        kept: List[ContentBlock] = []
        for block in blocks or []:
            mime_type, category = self._extract_block_metadata(block)
            if self._block_supported(mime_type, category):
                kept.append(block)
            else:
                removed.append(
                    _RemovedBlock(
                        category=category,
                        mime_type=mime_type,
                        source=source,
                        tool_id=tool_id,
                        block=block,
                    )
                )
        return kept

    def _block_supported(self, mime_type: str | None, category: str) -> bool:
        """Determine if the current model can process a content block."""
        if category == "text":
            return True

        model_name = self._llm.model_name if self._llm else None
        if not model_name:
            return False

        if mime_type:
            return ModelDatabase.supports_mime(model_name, mime_type)

        if category == "vision":
            return ModelDatabase.supports_any_mime(
                model_name, ["image/jpeg", "image/png", "image/webp"]
            )

        if category == "document":
            return ModelDatabase.supports_mime(model_name, "application/pdf")

        return False

    def _extract_block_metadata(self, block: ContentBlock) -> Tuple[str | None, str]:
        """Infer the MIME type and high-level category for a content block."""
        if isinstance(block, TextContent):
            return "text/plain", "text"

        if isinstance(block, TextResourceContents):
            mime = getattr(block, "mimeType", None) or "text/plain"
            return mime, "text"

        if isinstance(block, ImageContent):
            mime = getattr(block, "mimeType", None) or "image/*"
            return mime, "vision"

        if isinstance(block, EmbeddedResource):
            resource = getattr(block, "resource", None)
            mime = getattr(resource, "mimeType", None)
            if isinstance(resource, TextResourceContents) or (mime and is_text_mime_type(mime)):
                return mime or "text/plain", "text"
            if mime and mime.startswith("image/"):
                return mime, "vision"
            return mime, "document"

        if isinstance(block, ResourceLink):
            mime = getattr(block, "mimeType", None)
            if mime and mime.startswith("image/"):
                return mime, "vision"
            if mime and is_text_mime_type(mime):
                return mime, "text"
            return mime, "document"

        return None, "document"

    def _build_error_channel_entries(self, removed: List[_RemovedBlock]) -> List[ContentBlock]:
        """Create informative entries for the error channel."""
        entries: List[ContentBlock] = []
        model_name = self._llm.model_name if self._llm else None
        model_display = model_name or "current model"

        for item in removed:
            mime_display = item.mime_type or "unknown"
            category_label = self._category_label(item.category)
            if item.source == "message":
                source_label = "user content"
            elif item.tool_id:
                source_label = f"tool result '{item.tool_id}'"
            else:
                source_label = "tool result"

            message = (
                f"Removed unsupported {category_label} {source_label} ({mime_display}) "
                f"before sending to {model_display}."
            )
            entries.append(text_content(message))
            entries.append(item.block)

        return entries

    def _build_metadata_entries(self, removed: List[_RemovedBlock]) -> List[ContentBlock]:
        entries: List[ContentBlock] = []
        for item in removed:
            metadata_text = text_content(
                json.dumps(
                    {
                        "type": "fast-agent-removed",
                        "category": item.category,
                        "mime_type": item.mime_type,
                        "source": item.source,
                        "tool_id": item.tool_id,
                    }
                )
            )
            entries.append(metadata_text)
        return entries

    def _build_removed_summary(self, removed: List[_RemovedBlock]) -> RemovedContentSummary | None:
        if not removed:
            return None

        counts = Counter(item.category for item in removed)
        category_mimes: Dict[str, Tuple[str, ...]] = {}
        mime_accumulator: Dict[str, set[str]] = defaultdict(set)

        for item in removed:
            mime_accumulator[item.category].add(item.mime_type or "unknown")

        for category, mimes in mime_accumulator.items():
            category_mimes[category] = tuple(sorted(mimes))

        alert_flags = frozenset(
            flag
            for category in counts
            for flag in (self._category_to_flag(category),)
            if flag is not None
        )

        model_name = self._llm.model_name if self._llm else None
        model_display = model_name or "current model"

        category_order = ["vision", "document", "other", "text"]
        segments: List[str] = []
        for category in category_order:
            if category not in counts:
                continue
            count = counts[category]
            mime_list = ", ".join(category_mimes.get(category, ()))
            label = self._category_label(category)
            plural = "s" if count != 1 else ""
            if mime_list:
                segments.append(f"{count} {label} block{plural} ({mime_list})")
            else:
                segments.append(f"{count} {label} block{plural}")

        # Append any remaining categories not covered in the preferred order
        for category, count in counts.items():
            if category in category_order:
                continue
            mime_list = ", ".join(category_mimes.get(category, ()))
            label = self._category_label(category)
            plural = "s" if count != 1 else ""
            if mime_list:
                segments.append(f"{count} {label} block{plural} ({mime_list})")
            else:
                segments.append(f"{count} {label} block{plural}")

        detail = "; ".join(segments) if segments else "unknown content"

        capability_labels = []
        for flag in alert_flags:
            match flag:
                case "V":
                    capability_labels.append("vision")
                case "D":
                    capability_labels.append("document")
                case "T":
                    capability_labels.append("text")

        capability_note = ""
        if capability_labels:
            unique_caps = ", ".join(sorted(set(capability_labels)))
            capability_note = f" Missing capability: {unique_caps}."

        message = (
            f"Removed unsupported content before sending to {model_display}: {detail}."
            f"{capability_note} Stored original content in '{FAST_AGENT_ERROR_CHANNEL}'."
        )

        return RemovedContentSummary(
            model_name=model_name,
            counts=dict(counts),
            category_mimes=category_mimes,
            alert_flags=alert_flags,
            message=message,
        )

    @staticmethod
    def _category_to_flag(category: str) -> str | None:
        mapping = {"text": "T", "document": "D", "vision": "V"}
        return mapping.get(category)

    @staticmethod
    def _category_label(category: str) -> str:
        if category == "vision":
            return "vision"
        if category == "document":
            return "document"
        if category == "text":
            return "text"
        return "content"

    @property
    def message_history(self) -> List[PromptMessageExtended]:
        """
        Return the agent's message history as PromptMessageExtended objects.

        This history can be used to transfer state between agents or for
        analysis and debugging purposes.

        Returns:
            List of PromptMessageExtended objects representing the conversation history
        """
        if self._llm:
            return self._llm.message_history
        return []

    def pop_last_message(self) -> PromptMessageExtended | None:
        """Remove and return the most recent message from the conversation history."""
        if self._llm:
            return self._llm.pop_last_message()
        return None

    @property
    def usage_accumulator(self) -> UsageAccumulator | None:
        """
        Return the usage accumulator for tracking token usage across turns.

        Returns:
            UsageAccumulator object if LLM is attached, None otherwise
        """
        if self._llm:
            return self._llm.usage_accumulator
        return None

    @property
    def llm(self) -> FastAgentLLMProtocol:
        assert self._llm, "LLM is not attached"
        return self._llm

    # --- Default MCP-facing convenience methods (no-op for plain LLM agents) ---

    async def list_prompts(self, namespace: str | None = None) -> Mapping[str, List[Prompt]]:
        """Default: no prompts; return empty mapping."""
        return {}

    async def get_prompt(
        self,
        prompt_name: str,
        arguments: Dict[str, str] | None = None,
        namespace: str | None = None,
    ) -> GetPromptResult:
        """Default: prompts unsupported; return empty GetPromptResult."""
        return GetPromptResult(description="", messages=[])

    async def list_resources(self, namespace: str | None = None) -> Mapping[str, List[str]]:
        """Default: no resources; return empty mapping."""
        return {}

    async def list_tools(self) -> ListToolsResult:
        """Default: no tools; return empty ListToolsResult."""
        return ListToolsResult(tools=[])

    async def list_mcp_tools(self, namespace: str | None = None) -> Mapping[str, List[Tool]]:
        """Default: no tools; return empty mapping."""
        return {}

    async def get_resource(
        self, resource_uri: str, namespace: str | None = None
    ) -> ReadResourceResult:
        """Default: resources unsupported; raise capability error."""
        raise NotImplementedError("Resources are not supported by this agent")

    async def with_resource(
        self,
        prompt_content: Union[str, PromptMessage, PromptMessageExtended],
        resource_uri: str,
        namespace: str | None = None,
    ) -> str:
        """Default: ignore resource, just send the prompt content."""
        return await self.send(prompt_content)

    @property
    def provider(self) -> Provider:
        return self.llm.provider

    def _merge_request_params(
        self,
        base_params: RequestParams | None,
        override_params: RequestParams | None,
        model_override: str | None = None,
    ) -> RequestParams | None:
        """
        Merge request parameters with proper precedence.

        Args:
            base_params: Base parameters (lower precedence)
            override_params: Override parameters (higher precedence)
            model_override: Optional model name to override

        Returns:
            Merged RequestParams or None if both inputs are None
        """
        if not base_params and not override_params:
            return None

        if not base_params:
            result = override_params.model_copy() if override_params else None
        else:
            result = base_params.model_copy()
            if override_params:
                # Merge only the explicitly set values from override_params
                for k, v in override_params.model_dump(exclude_unset=True).items():
                    if v is not None:
                        setattr(result, k, v)

        # Apply model override if specified
        if model_override and result:
            result.model = model_override

        return result

    async def agent_card(self) -> AgentCard:
        """
        Return an A2A card describing this Agent
        """
        from fast_agent.agents.llm_agent import DEFAULT_CAPABILITIES

        return AgentCard(
            skills=[],
            name=self._name,
            description=self.instruction,
            url=f"fast-agent://agents/{self._name}/",
            version="0.1",
            capabilities=DEFAULT_CAPABILITIES,
            # TODO -- get these from the _llm
            default_input_modes=["text/plain"],
            default_output_modes=["text/plain"],
            provider=None,
            documentation_url=None,
        )

    async def run_tools(self, request: PromptMessageExtended) -> PromptMessageExtended:
        return request

    async def show_assistant_message(
        self,
        message: PromptMessageExtended,
        bottom_items: List[str] | None = None,
        highlight_items: str | List[str] | None = None,
        max_item_length: int | None = None,
        name: str | None = None,
        model: str | None = None,
        additional_message: Optional["Text"] = None,
    ) -> None:
        pass
