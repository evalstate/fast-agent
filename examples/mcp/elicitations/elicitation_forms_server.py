"""MCPServer form-elicitation demo using modern resolver APIs."""

from typing import Annotated, TypedDict, cast

from mcp.server.mcpserver import (
    AcceptedElicitation,
    CancelledElicitation,
    DeclinedElicitation,
    Elicit,
    ElicitationResult,
    MCPServer,
    Resolve,
)
from pydantic import BaseModel, Field


class TitledEnumOption(TypedDict):
    const: str
    title: str


def enum_schema_options(data: dict[str, str]) -> list[TitledEnumOption]:
    return [cast("TitledEnumOption", {"const": key, "title": value}) for key, value in data.items()]


WORKSHOPS = {
    "ai_basics": "AI Fundamentals",
    "llm_apps": "Building LLM Applications",
    "prompt_eng": "Prompt Engineering",
    "rag_systems": "RAG Systems",
    "fine_tuning": "Model Fine-tuning",
    "deployment": "Production Deployment",
}
CATEGORIES = {
    "electronics": "Electronics",
    "books": "Books & Media",
    "clothing": "Clothing",
    "home": "Home & Garden",
    "sports": "Sports & Outdoors",
}
THEMES = {"light": "Light Theme", "dark": "Dark Theme", "auto": "Auto (System)"}


class EventRegistration(BaseModel):
    name: str = Field(description="Your full name", min_length=2, max_length=100)
    email: str = Field(description="Your email address", json_schema_extra={"format": "email"})
    company_website: str = Field(
        "", description="Your company website (optional)", json_schema_extra={"format": "uri"}
    )
    workshops: list[str] = Field(
        description="Select the workshops you want to attend",
        min_length=1,
        max_length=3,
        json_schema_extra={
            "items": {"enum": list(WORKSHOPS), "enumNames": list(WORKSHOPS.values())},
            "uniqueItems": True,
        },
    )
    event_date: str = Field(
        description="Which event date works for you?", json_schema_extra={"format": "date"}
    )
    dietary_requirements: str = Field(
        "", description="Any dietary requirements? (optional)", max_length=200
    )


class ProductReview(BaseModel):
    rating: int = Field(description="Rate this product (1-5 stars)", ge=1, le=5)
    satisfaction: float = Field(
        description="Overall satisfaction score (0.0-10.0)", ge=0.0, le=10.0
    )
    category: str = Field(
        description="What type of product is this?",
        json_schema_extra={"oneOf": enum_schema_options(CATEGORIES)},
    )
    review_text: str = Field(
        default="Great product!",
        description="Tell us about your experience",
        min_length=10,
        max_length=1000,
    )


class AccountSettings(BaseModel):
    email_notifications: bool = Field(True, description="Receive email notifications?")
    marketing_emails: bool = Field(False, description="Subscribe to marketing emails?")
    theme: str = Field(
        "dark",
        description="Choose your theme",
        json_schema_extra={"oneOf": enum_schema_options(THEMES)},
    )
    privacy_public: bool = Field(False, description="Make your profile public?")
    items_per_page: int = Field(25, description="Items to show per page (10-100)", ge=10, le=100)


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
        description="Preferred appointment date and time", json_schema_extra={"format": "date-time"}
    )
    priority_service: bool = Field(False, description="Is this an urgent repair?")


def request_event_registration() -> Elicit[EventRegistration]:
    return Elicit(
        "Register for the fast-agent conference - fill out your details", EventRegistration
    )


def request_product_review() -> Elicit[ProductReview]:
    return Elicit("Share your product review - Help others make informed decisions!", ProductReview)


def request_account_settings() -> Elicit[AccountSettings]:
    return Elicit("Update your account settings", AccountSettings)


def request_service_appointment() -> Elicit[ServiceAppointment]:
    return Elicit("Schedule your vehicle service appointment", ServiceAppointment)


server = MCPServer("Elicitation Forms Demo Server")


@server.tool()
def event_registration(
    result: Annotated[ElicitationResult[EventRegistration], Resolve(request_event_registration)],
) -> str:
    """Register for a tech conference event."""
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
            lines.extend(f"   • {WORKSHOPS.get(workshop, workshop)}" for workshop in data.workshops)
            return "\n".join(lines)
        case DeclinedElicitation():
            return "Registration declined - no ticket reserved"
        case CancelledElicitation():
            return "Registration cancelled - please try again later"
    raise AssertionError(f"Unexpected elicitation result: {result}")


@server.tool()
def product_review(
    result: Annotated[ElicitationResult[ProductReview], Resolve(request_product_review)],
) -> str:
    """Submit a product review with rating and comments."""
    match result:
        case AcceptedElicitation(data=data):
            return "\n".join(
                [
                    "🎯 Product Review Submitted!",
                    f"⭐ Rating: {'⭐' * data.rating} ({data.rating}/5)",
                    f"📊 Satisfaction: {data.satisfaction}/10.0",
                    f"📦 Category: {CATEGORIES.get(data.category, data.category)}",
                    f"💬 Review: {data.review_text}",
                ]
            )
        case DeclinedElicitation():
            return "Review declined - no feedback submitted"
        case CancelledElicitation():
            return "Review cancelled - you can submit it later"
    raise AssertionError(f"Unexpected elicitation result: {result}")


@server.tool()
def account_settings(
    result: Annotated[ElicitationResult[AccountSettings], Resolve(request_account_settings)],
) -> str:
    """Configure account settings and preferences."""
    match result:
        case AcceptedElicitation(data=data):
            return "\n".join(
                [
                    "⚙️ Account Settings Updated!",
                    f"📧 Email notifications: {'On' if data.email_notifications else 'Off'}",
                    f"📬 Marketing emails: {'On' if data.marketing_emails else 'Off'}",
                    f"🎨 Theme: {THEMES.get(data.theme, data.theme)}",
                    f"👥 Public profile: {'Yes' if data.privacy_public else 'No'}",
                    f"📄 Items per page: {data.items_per_page}",
                ]
            )
        case DeclinedElicitation():
            return "Settings unchanged - keeping current preferences"
        case CancelledElicitation():
            return "Settings update cancelled"
    raise AssertionError(f"Unexpected elicitation result: {result}")


@server.tool()
def service_appointment(
    result: Annotated[ElicitationResult[ServiceAppointment], Resolve(request_service_appointment)],
) -> str:
    """Schedule a car service appointment."""
    match result:
        case AcceptedElicitation(data=data):
            return "\n".join(
                [
                    "🔧 Service Appointment Scheduled!",
                    f"👤 Customer: {data.customer_name}",
                    f"📞 Phone: {data.phone_number}",
                    f"🚗 Vehicle: {data.vehicle_type.title()}",
                    f"🚙 Loaner needed: {'Yes' if data.needs_loaner else 'No'}",
                    f"📅 Appointment: {data.appointment_time}",
                    f"⚡ Priority service: {'Yes' if data.priority_service else 'No'}",
                ]
            )
        case DeclinedElicitation():
            return "Appointment cancelled - call us when you're ready!"
        case CancelledElicitation():
            return "Appointment scheduling cancelled"
    raise AssertionError(f"Unexpected elicitation result: {result}")


if __name__ == "__main__":
    server.run(transport="stdio")
