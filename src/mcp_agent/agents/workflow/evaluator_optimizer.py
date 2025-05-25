"""
Evaluator-Optimizer workflow implementation using the BaseAgent adapter pattern.

This workflow provides a mechanism for iterative refinement of responses through
evaluation and feedback cycles. It uses one agent to generate responses and another
to evaluate and provide feedback, continuing until a quality threshold is reached
or a maximum number of refinements is attempted.
"""

from enum import Enum
from typing import Any, List, Optional, Tuple, Type

from pydantic import BaseModel, Field

from mcp_agent.agents.agent import Agent
from mcp_agent.agents.base_agent import BaseAgent
from mcp_agent.core.agent_types import AgentType
from mcp_agent.core.exceptions import AgentConfigError
from mcp_agent.core.prompt import Prompt
from mcp_agent.core.request_params import RequestParams
from mcp_agent.logging.logger import get_logger
from mcp_agent.mcp.interfaces import ModelT
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart

logger = get_logger(__name__)


class QualityRating(str, Enum):
    """Enum for evaluation quality ratings."""

    POOR = "POOR"  # Major improvements needed
    FAIR = "FAIR"  # Several improvements needed
    GOOD = "GOOD"  # Minor improvements possible
    EXCELLENT = "EXCELLENT"  # No improvements needed

    # Map string values to integer values for comparisons
    @property
    def value(self) -> int:
        """Convert string enum values to integers for comparison."""
        return {
            "POOR": 0,
            "FAIR": 1,
            "GOOD": 2,
            "EXCELLENT": 3,
        }[self._value_]


class EvaluationResult(BaseModel):
    """Model representing the evaluation result from the evaluator agent."""

    rating: QualityRating = Field(description="Quality rating of the response")
    feedback: str = Field(description="Specific feedback and suggestions for improvement")
    needs_improvement: bool = Field(description="Whether the output needs further improvement")
    focus_areas: List[str] = Field(
        default_factory=list, description="Specific areas to focus on in next iteration"
    )


class EvaluatorOptimizerAgent(BaseAgent):
    """
    An agent that implements the evaluator-optimizer workflow pattern.

    Uses one agent to generate responses and another to evaluate and provide feedback
    for refinement, continuing until a quality threshold is reached or a maximum
    number of refinement cycles is completed.
    """

    @property
    def agent_type(self) -> AgentType:
        """Return the type of this agent."""
        return AgentType.EVALUATOR_OPTIMIZER

    def __init__(
        self,
        config: Agent,
        generator_agent: Agent,
        evaluator_agent: Agent,
        min_rating: QualityRating = QualityRating.GOOD,
        max_refinements: int = 3,
        context: Optional[Any] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the evaluator-optimizer agent.

        Args:
            config: Agent configuration or name
            generator_agent: Agent that generates the initial and refined responses
            evaluator_agent: Agent that evaluates responses and provides feedback
            min_rating: Minimum acceptable quality rating to stop refinement
            max_refinements: Maximum number of refinement cycles to attempt
            context: Optional context object
            **kwargs: Additional keyword arguments to pass to BaseAgent
        """
        super().__init__(config, context=context, **kwargs)

        if not generator_agent:
            raise AgentConfigError("Generator agent must be provided")

        if not evaluator_agent:
            raise AgentConfigError("Evaluator agent must be provided")

        self.generator_agent = generator_agent
        self.evaluator_agent = evaluator_agent
        self.min_rating = min_rating
        self.max_refinements = max_refinements
        self.refinement_history = []

    async def generate(
        self,
        multipart_messages: List[PromptMessageMultipart],
        request_params: Optional[RequestParams] = None,
    ) -> PromptMessageMultipart:
        """
        Generate a response through evaluation-guided refinement.

        Args:
            multipart_messages: Messages to process
            request_params: Optional request parameters

        Returns:
            The optimized response after evaluation and refinement
        """
        # Initialize tracking variables
        refinement_count = 0
        best_response = None
        best_rating = QualityRating.POOR
        self.refinement_history = []

        # Extract the user request
        request = multipart_messages[-1].all_text() if multipart_messages else ""

        # Initial generation
        response = await self.generator_agent.generate(multipart_messages, request_params)
        best_response = response

        # Refinement loop
        while refinement_count < self.max_refinements:
            logger.debug(f"Evaluating response (iteration {refinement_count + 1})")

            # Evaluate current response
            eval_prompt = self._build_eval_prompt(
                request=request, response=response.all_text(), iteration=refinement_count
            )

            # Create evaluation message and get structured evaluation result
            eval_message = Prompt.user(eval_prompt)
            evaluation_result, _ = await self.evaluator_agent.structured(
                [eval_message], EvaluationResult, request_params
            )

            # If structured parsing failed, use default evaluation
            if evaluation_result is None:
                logger.warning("Structured parsing failed, using default evaluation")
                evaluation_result = EvaluationResult(
                    rating=QualityRating.POOR,
                    feedback="Failed to parse evaluation",
                    needs_improvement=True,
                    focus_areas=["Improve overall quality"],
                )

            # Track iteration
            self.refinement_history.append(
                {
                    "attempt": refinement_count + 1,
                    "response": response.all_text(),
                    "evaluation": evaluation_result.model_dump(),
                }
            )

            logger.debug(f"Evaluation result: {evaluation_result.rating}")

            # Track best response based on rating
            if evaluation_result.rating.value > best_rating.value:
                best_rating = evaluation_result.rating
                best_response = response
                logger.debug(f"New best response (rating: {best_rating})")

            # Check if we've reached acceptable quality
            if not evaluation_result.needs_improvement:
                logger.debug("Improvement not needed, stopping refinement")
                # When evaluator says no improvement needed, use the current response
                best_response = response
                break

            if evaluation_result.rating.value >= self.min_rating.value:
                logger.debug(f"Acceptable quality reached ({evaluation_result.rating})")
                break

            # Generate refined response
            refinement_prompt = self._build_refinement_prompt(
                request=request,
                response=response.all_text(),
                feedback=evaluation_result,
                iteration=refinement_count,
            )

            # Create refinement message and get refined response
            refinement_message = Prompt.user(refinement_prompt)
            response = await self.generator_agent.generate([refinement_message], request_params)

            refinement_count += 1

        return best_response

    async def structured(
        self,
        prompt: List[PromptMessageMultipart],
        model: Type[ModelT],
        request_params: Optional[RequestParams] = None,
    ) -> Tuple[ModelT | None, PromptMessageMultipart]:
        """
        Generate an optimized response and parse it into a structured format.

        Args:
            prompt: List of messages to process
            model: Pydantic model to parse the response into
            request_params: Optional request parameters

        Returns:
            The parsed response, or None if parsing fails
        """
        # Generate optimized response
        response = await self.generate(prompt, request_params)

        # Delegate structured parsing to the generator agent
        structured_prompt = Prompt.user(response.all_text())
        return await self.generator_agent.structured([structured_prompt], model, request_params)

    async def initialize(self) -> None:
        """Initialize the agent and its generator and evaluator agents."""
        await super().initialize()

        # Initialize generator and evaluator agents if not already initialized
        if not getattr(self.generator_agent, "initialized", False):
            await self.generator_agent.initialize()

        if not getattr(self.evaluator_agent, "initialized", False):
            await self.evaluator_agent.initialize()

        self.initialized = True

    async def shutdown(self) -> None:
        """Shutdown the agent and its generator and evaluator agents."""
        await super().shutdown()

        # Shutdown generator and evaluator agents
        try:
            await self.generator_agent.shutdown()
        except Exception as e:
            logger.warning(f"Error shutting down generator agent: {str(e)}")

        try:
            await self.evaluator_agent.shutdown()
        except Exception as e:
            logger.warning(f"Error shutting down evaluator agent: {str(e)}")

    def _build_eval_prompt(self, request: str, response: str, iteration: int) -> str:
        """
        Build the evaluation prompt for the evaluator agent.

        Args:
            request: The original user request
            response: The current response to evaluate
            iteration: The current iteration number

        Returns:
            Formatted evaluation prompt
        """
        return f"""
You are an expert evaluator for content quality. Your task is to evaluate a response against the user's original request.

Evaluate the response for iteration {iteration + 1} and provide structured feedback on its quality and areas for improvement.

<fastagent:data>
<fastagent:request>
{request}
</fastagent:request>

<fastagent:response>
{response}
</fastagent:response>
</fastagent:data>

<fastagent:instruction>
Your response MUST be valid JSON matching this exact format (no other text, markdown, or explanation):

{{
  "rating": "RATING",
  "feedback": "DETAILED FEEDBACK",
  "needs_improvement": BOOLEAN,
  "focus_areas": ["FOCUS_AREA_1", "FOCUS_AREA_2", "FOCUS_AREA_3"]
}}

Where:
- RATING: Must be one of: "EXCELLENT", "GOOD", "FAIR", or "POOR"
  - EXCELLENT: No improvements needed
  - GOOD: Only minor improvements possible
  - FAIR: Several improvements needed
  - POOR: Major improvements needed
- DETAILED FEEDBACK: Specific, actionable feedback (as a single string)
- BOOLEAN: true or false (lowercase, no quotes) indicating if further improvement is needed
- FOCUS_AREAS: Array of 1-3 specific areas to focus on (empty array if no improvement needed)

Example of valid response (DO NOT include the triple backticks in your response):
{{
  "rating": "GOOD",
  "feedback": "The response is clear but could use more supporting evidence.",
  "needs_improvement": true,
  "focus_areas": ["Add more examples", "Include data points"]
}}

IMPORTANT: Your response should be ONLY the JSON object without any code fences, explanations, or other text.
</fastagent:instruction>
"""

    def _build_refinement_prompt(
        self,
        request: str,
        response: str,
        feedback: EvaluationResult,
        iteration: int,
    ) -> str:
        """
        Build the refinement prompt for the generator agent.

        Args:
            request: The original user request
            response: The current response to refine
            feedback: The evaluation feedback
            iteration: The current iteration number

        Returns:
            Formatted refinement prompt
        """
        focus_areas = ", ".join(feedback.focus_areas) if feedback.focus_areas else "None specified"

        return f"""
You are tasked with improving a response based on expert feedback. This is iteration {iteration + 1} of the refinement process.

Your goal is to address all feedback points while maintaining accuracy and relevance to the original request.

<fastagent:data>
<fastagent:request>
{request}
</fastagent:request>

<fastagent:previous-response>
{response}
</fastagent:previous-response>

<fastagent:feedback>
<rating>{feedback.rating}</rating>
<details>{feedback.feedback}</details>
<focus-areas>{focus_areas}</focus-areas>
</fastagent:feedback>
</fastagent:data>

<fastagent:instruction>
Create an improved version of the response that:
1. Directly addresses each point in the feedback
2. Focuses on the specific areas mentioned for improvement
3. Maintains all the strengths of the original response
4. Remains accurate and relevant to the original request

Provide your complete improved response without explanations or commentary.
</fastagent:instruction>
"""
