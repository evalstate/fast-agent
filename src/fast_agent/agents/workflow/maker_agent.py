"""
MAKER: Massively decomposed Agentic processes with K-voting Error Reduction.

Implementation based on the paper:
"Solving a Million-Step LLM Task with Zero Errors" (arXiv:2511.09030)
https://arxiv.org/abs/2511.09030

This workflow implements first-to-ahead-by-k voting for statistical error
correction, enabling high reliability with cost-effective models. The key
insight is that by sampling multiple responses and requiring a k-vote margin
for consensus, the probability of error decreases exponentially while cost
grows only logarithmically with the number of steps.

Key concepts from the paper:
- Maximal Agentic Decomposition (MAD): Break tasks into single-step subtasks
- First-to-ahead-by-k voting: Winner needs k more votes than runner-up
- Red-flagging: Discard suspicious outputs (too long, malformed) before voting
"""

from collections import defaultdict
from enum import StrEnum
from typing import Any, Callable, List, Optional, Tuple, Type

from mcp import Tool
from opentelemetry import trace
from pydantic import BaseModel, Field

from fast_agent.agents.agent_types import AgentConfig, AgentType
from fast_agent.agents.llm_agent import LlmAgent
from fast_agent.core.exceptions import AgentConfigError
from fast_agent.core.logging.logger import get_logger
from fast_agent.core.prompt import Prompt
from fast_agent.interfaces import AgentProtocol, ModelT
from fast_agent.types import PromptMessageExtended, RequestParams

logger = get_logger(__name__)


class MatchStrategy(StrEnum):
    """
    Strategies for comparing responses during voting.

    The choice of strategy affects how responses are grouped for voting:
    - EXACT: Responses must match character-for-character
    - NORMALIZED: Whitespace and case differences are ignored
    - STRUCTURED: JSON responses are parsed and compared structurally
    """

    EXACT = "exact"
    NORMALIZED = "normalized"
    STRUCTURED = "structured"


class MakerResult(BaseModel):
    """
    Result of a MAKER voting process.

    Provides transparency into the voting outcome for debugging and analysis.
    """

    winner: str = Field(description="The winning response text")
    votes: dict[str, int] = Field(
        default_factory=dict, description="Vote counts per unique response"
    )
    total_samples: int = Field(default=0, description="Total samples drawn")
    discarded_samples: int = Field(
        default=0, description="Samples discarded due to red-flags"
    )
    margin: int = Field(default=0, description="Winning margin achieved")
    converged: bool = Field(
        default=False, description="Whether k-margin consensus was achieved"
    )


class MakerAgent(LlmAgent):
    """
    MAKER: Massively decomposed Agentic processes with K-voting Error Reduction.

    Implements first-to-ahead-by-k voting for statistical error correction.
    Multiple samples are drawn from a worker agent, and the first response
    to achieve a k-vote margin over all alternatives wins.

    This approach enables:
    - High reliability with cheap/small models
    - Logarithmic cost scaling with task complexity
    - Provable error bounds based on per-step success rate

    Reference: "Solving a Million-Step LLM Task with Zero Errors"
    https://arxiv.org/abs/2511.09030
    """

    @property
    def agent_type(self) -> AgentType:
        """Return the type of this agent."""
        return AgentType.MAKER

    def __init__(
        self,
        config: AgentConfig,
        worker_agent: AgentProtocol,
        k: int = 3,
        max_samples: int = 50,
        match_strategy: MatchStrategy = MatchStrategy.EXACT,
        match_fn: Callable[[str], str] | None = None,
        red_flag_max_length: int | None = None,
        red_flag_validator: Callable[[str], bool] | None = None,
        context: Optional[Any] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the MAKER agent.

        Args:
            config: Agent configuration
            worker_agent: The agent to sample from for voting
            k: Margin required to declare a winner (first-to-ahead-by-k).
               Higher k = more reliable but more samples needed.
               Paper recommends k >= 3 for high reliability.
            max_samples: Maximum samples before falling back to plurality vote
            match_strategy: How to compare responses for voting
            match_fn: Custom function to normalize responses for comparison.
                      If provided, overrides match_strategy.
            red_flag_max_length: Discard responses longer than this (characters).
                                 Per the paper, overly long responses correlate
                                 with errors.
            red_flag_validator: Custom validator function. Return False to
                                discard the response (red-flag it).
            context: Optional context object
        """
        super().__init__(config, context=context, **kwargs)

        if not worker_agent:
            raise AgentConfigError("Worker agent must be provided")
        if k < 1:
            raise AgentConfigError("k must be at least 1")
        if max_samples < k:
            raise AgentConfigError("max_samples must be at least k")

        self.worker_agent = worker_agent
        self.k = k
        self.max_samples = max_samples
        self.match_strategy = match_strategy
        self.match_fn = match_fn
        self.red_flag_max_length = red_flag_max_length
        self.red_flag_validator = red_flag_validator

        # Result tracking
        self.last_result: MakerResult | None = None

    def _normalize_response(self, response: str) -> str:
        """
        Normalize response for comparison based on configured strategy.

        Args:
            response: Raw response text

        Returns:
            Normalized response for vote counting
        """
        if self.match_fn:
            return self.match_fn(response)

        match self.match_strategy:
            case MatchStrategy.EXACT:
                return response
            case MatchStrategy.NORMALIZED:
                return " ".join(response.lower().split())
            case MatchStrategy.STRUCTURED:
                import json

                try:
                    parsed = json.loads(response)
                    return json.dumps(parsed, sort_keys=True)
                except json.JSONDecodeError:
                    return response

        return response

    def _is_red_flagged(self, response: str) -> bool:
        """
        Check if response should be discarded (red-flagged).

        Per the MAKER paper, red-flagging improves effective success rate
        by discarding responses that show signs of confusion:
        - Overly long responses (model went off track)
        - Malformed responses (parsing issues indicate confusion)

        Args:
            response: Response text to check

        Returns:
            True if response should be discarded
        """
        if self.red_flag_max_length and len(response) > self.red_flag_max_length:
            logger.debug(
                f"Red-flagged: response length {len(response)} > {self.red_flag_max_length}"
            )
            return True

        if self.red_flag_validator and not self.red_flag_validator(response):
            logger.debug("Red-flagged: custom validator returned False")
            return True

        return False

    def _check_winner(self, votes: dict[str, int]) -> str | None:
        """
        Check if any response has achieved k-margin victory.

        First-to-ahead-by-k: winner needs k more votes than the runner-up.

        Args:
            votes: Current vote counts

        Returns:
            Winning response key if k-margin achieved, None otherwise
        """
        if not votes:
            return None

        sorted_items = sorted(votes.items(), key=lambda x: x[1], reverse=True)
        leader_key, leader_votes = sorted_items[0]
        runner_up_votes = sorted_items[1][1] if len(sorted_items) > 1 else 0

        if leader_votes - runner_up_votes >= self.k:
            return leader_key

        return None

    async def generate_impl(
        self,
        messages: List[PromptMessageExtended],
        request_params: RequestParams | None = None,
        tools: List[Tool] | None = None,
    ) -> PromptMessageExtended:
        """
        Generate a response using first-to-ahead-by-k voting.

        Samples from the worker agent until one response achieves a k-vote
        margin over all alternatives, or max_samples is reached.

        Args:
            messages: Input messages
            request_params: Optional request parameters
            tools: Optional tools (passed to worker)

        Returns:
            The winning response
        """
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(f"Maker: '{self._name}' generate"):
            votes: dict[str, int] = defaultdict(int)
            response_map: dict[str, PromptMessageExtended] = {}
            total_samples = 0
            discarded_samples = 0

            while total_samples < self.max_samples:
                async with self.workflow_telemetry.start_step(
                    "maker.sample",
                    server_name=self.name,
                    arguments={
                        "agent": self.worker_agent.name,
                        "sample": total_samples + 1,
                        "current_votes": dict(votes),
                    },
                ) as step:
                    response = await self.worker_agent.generate(
                        messages, request_params
                    )
                    response_text = response.last_text() or ""
                    total_samples += 1

                    # Red-flag check
                    if self._is_red_flagged(response_text):
                        discarded_samples += 1
                        await step.finish(
                            False, text=f"Sample {total_samples} red-flagged, discarded"
                        )
                        continue

                    # Normalize and record vote
                    normalized = self._normalize_response(response_text)
                    votes[normalized] += 1
                    response_map[normalized] = response

                    await step.finish(
                        True,
                        text=f"Sample {total_samples}: {votes[normalized]} votes for this response",
                    )

                # Check for k-margin winner
                winner_key = self._check_winner(votes)
                if winner_key:
                    sorted_votes = sorted(votes.values(), reverse=True)
                    margin = sorted_votes[0] - (
                        sorted_votes[1] if len(sorted_votes) > 1 else 0
                    )

                    self.last_result = MakerResult(
                        winner=winner_key,
                        votes=dict(votes),
                        total_samples=total_samples,
                        discarded_samples=discarded_samples,
                        margin=margin,
                        converged=True,
                    )

                    logger.debug(
                        f"MAKER converged: {votes[winner_key]} votes, "
                        f"margin {margin}, {total_samples} samples"
                    )
                    return response_map[winner_key]

            # Max samples reached - fall back to plurality
            logger.warning(
                f"MAKER: max_samples ({self.max_samples}) reached without "
                f"k-margin ({self.k}) consensus, using plurality"
            )

            if not votes:
                # All samples were red-flagged
                raise AgentConfigError(
                    f"All {total_samples} samples were red-flagged. "
                    "Consider relaxing red-flag criteria."
                )

            winner_key = max(votes, key=lambda x: votes[x])
            sorted_votes = sorted(votes.values(), reverse=True)
            margin = sorted_votes[0] - (
                sorted_votes[1] if len(sorted_votes) > 1 else 0
            )

            self.last_result = MakerResult(
                winner=winner_key,
                votes=dict(votes),
                total_samples=total_samples,
                discarded_samples=discarded_samples,
                margin=margin,
                converged=False,
            )

            return response_map[winner_key]

    async def structured_impl(
        self,
        messages: List[PromptMessageExtended],
        model: Type[ModelT],
        request_params: RequestParams | None = None,
    ) -> Tuple[ModelT | None, PromptMessageExtended]:
        """
        Generate a voted response and parse into structured format.

        Args:
            messages: Input messages
            model: Pydantic model class for structured output
            request_params: Optional request parameters

        Returns:
            Tuple of (parsed model or None, raw response)
        """
        response = await self.generate_impl(messages, request_params)
        return await self.worker_agent.structured(
            [Prompt.user(response.all_text())], model, request_params
        )

    async def initialize(self) -> None:
        """Initialize the agent and its worker agent."""
        await super().initialize()
        if not self.worker_agent.initialized:
            await self.worker_agent.initialize()
        self.initialized = True

    async def shutdown(self) -> None:
        """Shutdown the agent and its worker agent."""
        await super().shutdown()
        try:
            await self.worker_agent.shutdown()
        except Exception as e:
            logger.warning(f"Error shutting down worker agent: {str(e)}")
