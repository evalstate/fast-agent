"""
MCP Server for Basic Elicitation Forms Demo

This server provides various elicitation resources that demonstrate
different form types and validation patterns.
"""

import logging
import sys
from typing import Optional, TypedDict

from mcp import ReadResourceResult
from mcp.server.elicitation import (
    AcceptedElicitation,
    CancelledElicitation,
    DeclinedElicitation,
)
from mcp.server.fastmcp import FastMCP
from mcp.types import TextResourceContents
from pydantic import AnyUrl, BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("elicitation_forms_server")

# Create MCP server
mcp = FastMCP("Elicitation Forms Demo Server", log_level="INFO")


class TitledEnumOption(TypedDict):
    """Type definition for oneOf/anyOf schema options."""

    const: str
    title: str


def _create_enum_schema_options(data: dict[str, str]) -> list[TitledEnumOption]:
    """Convert a dictionary to oneOf/anyOf schema format.

    Args:
        data: Dictionary mapping enum values to display titles

    Returns:
        List of schema options with 'const' and 'title' fields

    Example:
        >>> _create_enum_schema_options({"dark": "Dark Mode", "light": "Light Mode"})
        [{"const": "dark", "title": "Dark Mode"}, {"const": "light", "title": "Light Mode"}]
    """
    return [{"const": k, "title": v} for k, v in data.items()]


@mcp.resource(uri="elicitation://event-registration")
async def event_registration() -> ReadResourceResult:
    """Register for a tech conference event."""
    workshop_names = {
        "ai_basics": "AI Fundamentals",
        "llm_apps": "Building LLM Applications",
        "prompt_eng": "Prompt Engineering",
        "rag_systems": "RAG Systems",
        "fine_tuning": "Model Fine-tuning",
        "deployment": "Production Deployment",
    }

    class EventRegistration(BaseModel):
        name: str = Field(description="Your full name", min_length=2, max_length=100)
        email: str = Field(description="Your email address", json_schema_extra={"format": "email"})
        company_website: str | None = Field(
            None, description="Your company website (optional)", json_schema_extra={"format": "uri"}
        )
        workshops: list[str] = Field(
            description="Select the workshops you want to attend",
            min_length=1,
            max_length=3,
            json_schema_extra={
                "items": {
                    "enum": list(workshop_names.keys()),
                    "enumNames": list(workshop_names.values()),
                },
                "uniqueItems": True,
            },
        )
        event_date: str = Field(
            description="Which event date works for you?", json_schema_extra={"format": "date"}
        )
        dietary_requirements: Optional[str] = Field(
            None, description="Any dietary requirements? (optional)", max_length=200
        )

    result = await mcp.get_context().elicit(
        "Register for the fast-agent conference - fill out your details",
        schema=EventRegistration,
    )

    match result:
        case AcceptedElicitation(data=data):
            lines = [
                f"✅ Registration confirmed for {data.name}",
                f"📧 Email: {data.email}",
                f"🏢 Company: {data.company_website or 'Not provided'}",
                f"📅 Event Date: {data.event_date}",
                f"🍽️ Dietary Requirements: {data.dietary_requirements or 'None'}",
                f"🎓 Workshops ({len(data.workshops)} selected):",
            ]
            for workshop in data.workshops:
                lines.append(f"   • {workshop_names.get(workshop, workshop)}")
            response = "\n".join(lines)
        case DeclinedElicitation():
            response = "Registration declined - no ticket reserved"
        case CancelledElicitation():
            response = "Registration cancelled - please try again later"

    return ReadResourceResult(
        contents=[
            TextResourceContents(
                mimeType="text/plain", uri=AnyUrl("elicitation://event-registration"), text=response
            )
        ]
    )


@mcp.resource(uri="elicitation://product-review")
async def product_review() -> ReadResourceResult:
    """Submit a product review with rating and comments."""
    categories = {
        "electronics": "Electronics",
        "books": "Books & Media",
        "clothing": "Clothing",
        "home": "Home & Garden",
        "sports": "Sports & Outdoors",
    }

    class ProductReview(BaseModel):
        rating: int = Field(description="Rate this product (1-5 stars)", ge=1, le=5)
        satisfaction: float = Field(
            description="Overall satisfaction score (0.0-10.0)", ge=0.0, le=10.0
        )
        category: str = Field(
            description="What type of product is this?",
            json_schema_extra={"oneOf": _create_enum_schema_options(categories)},
        )
        review_text: str = Field(
            description="Tell us about your experience",
            default="""Great product!
Here's what I loved:

- Excellent build quality
- Fast shipping
- Works as advertised

One minor issue:
- Instructions could be clearer

Overall, highly recommended!""",
            min_length=10,
            max_length=1000,
        )

    result = await mcp.get_context().elicit(
        "Share your product review - Help others make informed decisions!",
        schema=ProductReview,
    )

    match result:
        case AcceptedElicitation(data=data):
            stars = "⭐" * data.rating
            lines = [
                "🎯 Product Review Submitted!",
                f"⭐ Rating: {stars} ({data.rating}/5)",
                f"📊 Satisfaction: {data.satisfaction}/10.0",
                f"📦 Category: {categories.get(data.category, data.category)}",
                f"💬 Review: {data.review_text}",
            ]
            response = "\n".join(lines)
        case DeclinedElicitation():
            response = "Review declined - no feedback submitted"
        case CancelledElicitation():
            response = "Review cancelled - you can submit it later"

    return ReadResourceResult(
        contents=[
            TextResourceContents(
                mimeType="text/plain", uri=AnyUrl("elicitation://product-review"), text=response
            )
        ]
    )


@mcp.resource(uri="elicitation://account-settings")
async def account_settings() -> ReadResourceResult:
    """Configure your account settings and preferences."""

    themes = {"light": "Light Theme", "dark": "Dark Theme", "auto": "Auto (System)"}

    class AccountSettings(BaseModel):
        email_notifications: bool = Field(True, description="Receive email notifications?")
        marketing_emails: bool = Field(False, description="Subscribe to marketing emails?")
        theme: str = Field(
            "dark",
            description="Choose your preferred theme",
            json_schema_extra={"oneOf": _create_enum_schema_options(themes)},
        )
        privacy_public: bool = Field(False, description="Make your profile public?")
        items_per_page: int = Field(
            25, description="Items to show per page (10-100)", ge=10, le=100
        )

    result = await mcp.get_context().elicit("Update your account settings", schema=AccountSettings)

    match result:
        case AcceptedElicitation(data=data):
            lines = [
                "⚙️ Account Settings Updated!",
                f"📧 Email notifications: {'On' if data.email_notifications else 'Off'}",
                f"📬 Marketing emails: {'On' if data.marketing_emails else 'Off'}",
                f"🎨 Theme: {themes.get(data.theme, data.theme)}",
                f"👥 Public profile: {'Yes' if data.privacy_public else 'No'}",
                f"📄 Items per page: {data.items_per_page}",
            ]
            response = "\n".join(lines)
        case DeclinedElicitation():
            response = "Settings unchanged - keeping current preferences"
        case CancelledElicitation():
            response = "Settings update cancelled"

    return ReadResourceResult(
        contents=[
            TextResourceContents(
                mimeType="text/plain", uri=AnyUrl("elicitation://account-settings"), text=response
            )
        ]
    )


@mcp.resource(uri="elicitation://service-appointment")
async def service_appointment() -> ReadResourceResult:
    """Schedule a car service appointment."""

    class ServiceAppointment(BaseModel):
        customer_name: str = Field(description="Your full name", min_length=2, max_length=50)
        phone_number: str = Field(
            "555-", description="Contact phone number", min_length=10, max_length=20
        )
        vehicle_type: str = Field(
            default="sedan",
            description="What type of vehicle do you have?",
            json_schema_extra={
                "enum": ["sedan", "suv", "truck", "motorcycle", "other"],
                "enumNames": ["Sedan", "SUV/Crossover", "Truck", "Motorcycle", "Other"],
            },
        )
        needs_loaner: bool = Field(description="Do you need a loaner vehicle?")
        appointment_time: str = Field(
            description="Preferred appointment date and time",
            json_schema_extra={"format": "date-time"},
        )
        priority_service: bool = Field(False, description="Is this an urgent repair?")

    result = await mcp.get_context().elicit(
        "Schedule your vehicle service appointment", schema=ServiceAppointment
    )

    match result:
        case AcceptedElicitation(data=data):
            lines = [
                "🔧 Service Appointment Scheduled!",
                f"👤 Customer: {data.customer_name}",
                f"📞 Phone: {data.phone_number}",
                f"🚗 Vehicle: {data.vehicle_type.title()}",
                f"🚙 Loaner needed: {'Yes' if data.needs_loaner else 'No'}",
                f"📅 Appointment: {data.appointment_time}",
                f"⚡ Priority service: {'Yes' if data.priority_service else 'No'}",
            ]
            response = "\n".join(lines)
        case DeclinedElicitation():
            response = "Appointment cancelled - call us when you're ready!"
        case CancelledElicitation():
            response = "Appointment scheduling cancelled"

    return ReadResourceResult(
        contents=[
            TextResourceContents(
                mimeType="text/plain",
                uri=AnyUrl("elicitation://service-appointment"),
                text=response,
            )
        ]
    )


if __name__ == "__main__":
    logger.info("Starting elicitation forms demo server...")
    mcp.run()
