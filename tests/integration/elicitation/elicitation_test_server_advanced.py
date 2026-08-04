"""Modern MCPServer fixture for form elicitation integration tests."""

from typing import Annotated

from mcp.server.mcpserver import (
    AcceptedElicitation,
    CancelledElicitation,
    Context,
    DeclinedElicitation,
    Elicit,
    ElicitationResult,
    MCPServer,
    Resolve,
)
from pydantic import BaseModel, Field


class ServerRating(BaseModel):
    rating: bool = Field(description="Do you like this server?")


class UserProfile(BaseModel):
    name: str = Field(description="Your full name", min_length=2, max_length=50)
    age: int = Field(description="Your age", ge=0, le=150)
    role: str = Field(
        description="Your job role",
        json_schema_extra={
            "enum": ["developer", "designer", "manager", "qa", "other"],
            "enumNames": [
                "Software Developer",
                "UI/UX Designer",
                "Project Manager",
                "Quality Assurance",
                "Other",
            ],
        },
    )
    email: str = Field(
        "", description="Your email address (optional)", json_schema_extra={"format": "email"}
    )
    subscribe_newsletter: bool = Field(False, description="Subscribe to our newsletter?")


class Preferences(BaseModel):
    theme: str = Field(
        description="Choose your preferred theme",
        json_schema_extra={
            "enum": ["light", "dark", "auto"],
            "enumNames": ["Light Theme", "Dark Theme", "Auto Theme"],
        },
    )
    language: str = Field(
        description="Select your language",
        json_schema_extra={
            "enum": ["en", "es", "fr", "de"],
            "enumNames": ["English", "Spanish", "French", "German"],
        },
    )
    notifications: bool = Field(True, description="Enable notifications?")


class Feedback(BaseModel):
    overall_rating: int = Field(description="Overall rating (1-5)", ge=1, le=5)
    ease_of_use: float = Field(description="Ease of use (0.0-10.0)", ge=0.0, le=10.0)
    would_recommend: bool = Field(description="Would you recommend to others?")
    comments: str = Field("", description="Additional comments", max_length=500)


def request_simple_rating() -> Elicit[ServerRating]:
    return Elicit("Please rate this server", ServerRating)


def request_user_profile() -> Elicit[UserProfile]:
    return Elicit("Please provide your user profile information", UserProfile)


def request_preferences() -> Elicit[Preferences]:
    return Elicit("Configure your preferences", Preferences)


def request_feedback() -> Elicit[Feedback]:
    return Elicit("We'd love your feedback!", Feedback)


server = MCPServer("MCP Advanced Elicitation Server")


@server.tool()
def client_capabilities(ctx: Context) -> str:
    """Expose the capabilities the modern client supplied for this call."""
    capabilities = ctx.client_capabilities
    if capabilities is None:
        return "No client capabilities available"

    capability_lines = [
        f"{'✓' if capabilities.elicitation is not None else '✗'} Elicitation",
        f"{'✓' if capabilities.sampling is not None else '✗'} Sampling",
        f"{'✓' if capabilities.roots is not None else '✗'} Roots",
    ]
    return "Client Capabilities:\n" + "\n".join(capability_lines)


@server.tool()
def simple_rating(
    result: Annotated[ElicitationResult[ServerRating], Resolve(request_simple_rating)],
) -> str:
    """Request a simple boolean rating."""
    match result:
        case AcceptedElicitation(data=data):
            assert isinstance(data, ServerRating)
            return f"You {'liked' if data.rating else 'did not like'} the server"
        case DeclinedElicitation():
            return "Rating declined"
        case CancelledElicitation():
            return "Rating cancelled"
    raise AssertionError(f"Unexpected elicitation result: {result}")


@server.tool()
def user_profile(
    result: Annotated[ElicitationResult[UserProfile], Resolve(request_user_profile)],
) -> str:
    """Request a complex user profile form."""
    match result:
        case AcceptedElicitation(data=data):
            assert isinstance(data, UserProfile)
            lines = [
                f"Name: {data.name}",
                f"Age: {data.age}",
                f"Role: {data.role.title()}",
                f"Email: {data.email or 'Not provided'}",
                f"Newsletter: {'Yes' if data.subscribe_newsletter else 'No'}",
            ]
            return "Profile received:\n" + "\n".join(lines)
        case DeclinedElicitation():
            return "Profile declined"
        case CancelledElicitation():
            return "Profile cancelled"
    raise AssertionError(f"Unexpected elicitation result: {result}")


@server.tool()
def preferences(
    result: Annotated[ElicitationResult[Preferences], Resolve(request_preferences)],
) -> str:
    """Request enum-based preferences."""
    match result:
        case AcceptedElicitation(data=data):
            assert isinstance(data, Preferences)
            return (
                "Preferences set: "
                f"Theme={data.theme}, Language={data.language}, Notifications={data.notifications}"
            )
        case DeclinedElicitation():
            return "Preferences declined"
        case CancelledElicitation():
            return "Preferences cancelled"
    raise AssertionError(f"Unexpected elicitation result: {result}")


@server.tool()
def feedback(
    result: Annotated[ElicitationResult[Feedback], Resolve(request_feedback)],
) -> str:
    """Request rating and feedback fields."""
    match result:
        case AcceptedElicitation(data=data):
            assert isinstance(data, Feedback)
            lines = [
                f"Overall: {data.overall_rating}/5",
                f"Ease of use: {data.ease_of_use}/10.0",
                f"Would recommend: {'Yes' if data.would_recommend else 'No'}",
            ]
            if data.comments:
                lines.append(f"Comments: {data.comments}")
            return "Feedback received:\n" + "\n".join(lines)
        case DeclinedElicitation():
            return "Feedback declined"
        case CancelledElicitation():
            return "Feedback cancelled"
    raise AssertionError(f"Unexpected elicitation result: {result}")


if __name__ == "__main__":
    server.run(transport="stdio")
