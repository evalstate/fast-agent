"""Local typed models for Agent Trajectory Interchange Format v1.7."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class AtifModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AtifToolCall(AtifModel):
    tool_call_id: str
    function_name: str
    arguments: dict[str, Any]
    extra: dict[str, Any] | None = None


class AtifImageSource(AtifModel):
    media_type: Literal["image/jpeg", "image/png", "image/gif", "image/webp"]
    path: str


class AtifContentPart(AtifModel):
    type: Literal["text", "image"]
    text: str | None = None
    source: AtifImageSource | None = None

    @model_validator(mode="after")
    def validate_content(self) -> Self:
        if self.type == "text" and (self.text is None or self.source is not None):
            raise ValueError("text parts require text and forbid source")
        if self.type == "image" and (self.source is None or self.text is not None):
            raise ValueError("image parts require source and forbid text")
        return self


AtifContent = str | list[AtifContentPart]


class AtifSubagentTrajectoryRef(AtifModel):
    trajectory_id: str | None = None
    session_id: str | None = None
    trajectory_path: str | None = None
    extra: dict[str, Any] | None = None

    @model_validator(mode="after")
    def validate_resolvable(self) -> Self:
        if self.trajectory_id is None and self.trajectory_path is None:
            raise ValueError("subagent references require trajectory_id or trajectory_path")
        return self


class AtifObservationResult(AtifModel):
    source_call_id: str | None = None
    content: AtifContent | None = None
    subagent_trajectory_ref: list[AtifSubagentTrajectoryRef] | None = None
    extra: dict[str, Any] | None = None


class AtifObservation(AtifModel):
    results: list[AtifObservationResult]


class AtifMetrics(AtifModel):
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cached_tokens: int | None = None
    cost_usd: float | None = None
    prompt_token_ids: list[int] | None = None
    completion_token_ids: list[int] | None = None
    logprobs: list[float] | None = None
    extra: dict[str, Any] | None = None


class AtifStep(AtifModel):
    step_id: int = Field(ge=1)
    timestamp: str | None = None
    source: Literal["system", "user", "agent"]
    model_name: str | None = None
    reasoning_effort: str | float | None = None
    message: AtifContent
    reasoning_content: str | None = None
    tool_calls: list[AtifToolCall] | None = None
    observation: AtifObservation | None = None
    metrics: AtifMetrics | None = None
    is_copied_context: bool | None = None
    llm_call_count: int | None = Field(default=None, ge=0)
    extra: dict[str, Any] | None = None

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp(cls, value: str | None) -> str | None:
        if value is not None:
            datetime.fromisoformat(value.replace("Z", "+00:00"))
        return value

    @model_validator(mode="after")
    def validate_step_semantics(self) -> Self:
        if self.source != "agent":
            agent_fields = (
                self.model_name,
                self.reasoning_effort,
                self.reasoning_content,
                self.tool_calls,
                self.metrics,
            )
            if any(value is not None for value in agent_fields):
                raise ValueError("LLM fields are only valid on agent steps")
        if self.source == "agent" and self.llm_call_count == 0:
            if self.metrics is not None or self.reasoning_content is not None:
                raise ValueError(
                    "metrics and reasoning_content must be absent when llm_call_count is 0"
                )
        return self


class AtifAgent(AtifModel):
    name: str
    version: str
    model_name: str | None = None
    tool_definitions: list[dict[str, object]] | None = None
    extra: dict[str, Any] | None = None


class AtifFinalMetrics(AtifModel):
    total_prompt_tokens: int | None = None
    total_completion_tokens: int | None = None
    total_cached_tokens: int | None = None
    total_cost_usd: float | None = None
    total_steps: int | None = Field(default=None, ge=0)
    extra: dict[str, Any] | None = None


class AtifTrajectory(AtifModel):
    schema_version: Literal["ATIF-v1.7"] = "ATIF-v1.7"
    session_id: str | None = None
    trajectory_id: str | None = None
    agent: AtifAgent
    steps: list[AtifStep] = Field(min_length=1)
    notes: str | None = None
    final_metrics: AtifFinalMetrics | None = None
    continued_trajectory_ref: str | None = None
    extra: dict[str, Any] | None = None
    subagent_trajectories: list[AtifTrajectory] | None = None

    @model_validator(mode="after")
    def validate_trajectory(self) -> Self:
        for index, step in enumerate(self.steps, start=1):
            if step.step_id != index:
                raise ValueError(f"step_id must be sequential from 1; expected {index}")
            call_ids = {call.tool_call_id for call in step.tool_calls or []}
            for result in step.observation.results if step.observation else []:
                if result.source_call_id is not None and result.source_call_id not in call_ids:
                    raise ValueError(
                        f"observation source_call_id {result.source_call_id!r} has no tool call"
                    )
        trajectory_ids: set[str] = set()
        for trajectory in self.subagent_trajectories or []:
            if trajectory.trajectory_id is None:
                raise ValueError("embedded subagent trajectory_id is required")
            if trajectory.trajectory_id in trajectory_ids:
                raise ValueError("embedded subagent trajectory_id must be unique")
            trajectory_ids.add(trajectory.trajectory_id)
        for step in self.steps:
            for result in step.observation.results if step.observation else []:
                for reference in result.subagent_trajectory_ref or []:
                    if (
                        reference.trajectory_path is None
                        and reference.trajectory_id not in trajectory_ids
                    ):
                        raise ValueError(
                            "embedded subagent reference trajectory_id must resolve "
                            "within subagent_trajectories"
                        )
        return self

    def to_json_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)
