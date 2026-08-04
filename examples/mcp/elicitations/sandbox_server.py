"""Modern, request-scoped MCP elicitation example."""

from decimal import Decimal
from typing import Annotated, Literal

from mcp.server.mcpserver import (
    AcceptedElicitation,
    CancelledElicitation,
    DeclinedElicitation,
    Elicit,
    ElicitationResult,
    MCPServer,
    Resolve,
)
from pydantic import BaseModel, Field, field_validator

HOURLY_RATE_CENTS = 40


def usd_to_cents(value: float) -> int:
    cents = Decimal(str(value)) * 100
    if cents != cents.to_integral_value():
        raise ValueError("USD amounts must use whole cents")
    return int(cents)


def format_usd(cents: int) -> str:
    dollars, remainder = divmod(cents, 100)
    return f"${dollars}.{remainder:02d}"


class SandboxRequest(BaseModel):
    region: Literal["us-east-1", "us-west-2", "eu-west-1"] = Field(
        "us-east-1",
        description="Region for the simulated sandbox",
    )
    duration_hours: int = Field(
        1,
        description="Reservation length in hours",
        ge=1,
        le=8,
    )

    max_budget_usd: float = Field(
        3.20,
        title="Maximum budget (USD)",
        description="Maximum simulated spend; no charge occurs",
        ge=0.40,
        le=3.20,
    )

    @field_validator("max_budget_usd")
    @classmethod
    def validate_whole_cents(cls, value: float) -> float:
        usd_to_cents(value)
        return value


def request_sandbox_details() -> Elicit[SandboxRequest]:
    return Elicit(
        "Start t4-small sandbox at $0.40 per hour?",
        SandboxRequest,
    )


server = MCPServer("t4-small Sandbox Simulator")


@server.tool()
def start_t4_small_sandbox(
    request: Annotated[
        ElicitationResult[SandboxRequest],
        Resolve(request_sandbox_details),
    ],
) -> str:
    """Simulate starting a t4-small sandbox without infrastructure or charges."""
    match request:
        case AcceptedElicitation(data=data):
            assert isinstance(data, SandboxRequest)
            maximum_cost_cents = data.duration_hours * HOURLY_RATE_CENTS
            budget_cents = usd_to_cents(data.max_budget_usd)
            if maximum_cost_cents > budget_cents:
                return (
                    f"SIMULATED: estimated maximum cost {format_usd(maximum_cost_cents)} "
                    f"exceeds maximum budget {format_usd(budget_cents)}; "
                    "no sandbox was created"
                )
            return (
                "SIMULATED: started sandbox sim-t4-small-001\n"
                f"Profile: t4-small (NVIDIA T4)\nRegion: {data.region}\n"
                f"Duration: {data.duration_hours} hour(s)\n"
                "Rate: $0.40 per hour\n"
                f"Maximum cost: {format_usd(maximum_cost_cents)}\n"
                f"Maximum budget: {format_usd(budget_cents)}\n"
                "No infrastructure or billing was created."
            )
        case DeclinedElicitation():
            return "SIMULATED: start declined; no sandbox was created"
        case CancelledElicitation():
            return "SIMULATED: start cancelled; no sandbox was created"
    raise AssertionError(f"Unexpected elicitation result: {request}")


if __name__ == "__main__":
    server.run(transport="stdio")
