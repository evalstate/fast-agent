"""
Decorator for LlmAgent, normalizes PromptMessageExtended, allows easy extension of Agents
"""

from typing import (
    TYPE_CHECKING,
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
from mcp import Tool
from mcp.types import (
    GetPromptResult,
    Prompt,
    PromptMessage,
    ReadResourceResult,
)
from opentelemetry import trace
from pydantic import BaseModel

from fast_agent.agents.agent_types import AgentConfig, AgentType
from fast_agent.context import Context
from fast_agent.core.logging.logger import get_logger
from fast_agent.interfaces import (
    AgentProtocol,
    FastAgentLLMProtocol,
    LLMFactoryProtocol,
)
from fast_agent.llm.provider_types import Provider
from fast_agent.llm.usage_tracking import UsageAccumulator
from fast_agent.mcp.helpers.content_helpers import normalize_to_extended_list
from fast_agent.types import PromptMessageExtended, RequestParams

logger = get_logger(__name__)
# Define a TypeVar for models
ModelT = TypeVar("ModelT", bound=BaseModel)

# Define a TypeVar for AugmentedLLM and its subclasses
LLM = TypeVar("LLM", bound=FastAgentLLMProtocol)


class LlmDecorator(AgentProtocol):
    """
    A pure delegation wrapper around LlmAgent instances.

    This class provides simple delegation to an attached LLM without adding
    any LLM interaction behaviors. Subclasses can add specialized logic
    for stop reason handling, UI display, tool execution, etc.
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

        with self._tracer.start_as_current_span(f"Agent: '{self._name}' generate"):
            return await self.generate_impl(multipart_messages, request_params, tools)

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
        assert self._llm, "LLM is not attached"
        return await self._llm.generate(messages, request_params, tools)

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

        with self._tracer.start_as_current_span(f"Agent: '{self._name}' structured"):
            return await self.structured_impl(multipart_messages, model, request_params)

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
        assert self._llm, "LLM is not attached"
        return await self._llm.structured(messages, model, request_params)

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
