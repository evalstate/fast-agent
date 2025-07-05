"""
Testing elicitation handlers for integration tests.

These handlers are designed specifically for testing scenarios
where you need predictable, automated responses.
"""

from typing import TYPE_CHECKING, Any, Dict

from mcp.shared.context import RequestContext
from mcp.types import ElicitRequestParams, ElicitResult

from mcp_agent.logging.logger import get_logger

if TYPE_CHECKING:
    from mcp import ClientSession

logger = get_logger(__name__)


async def auto_accept_test_handler(
    context: RequestContext["ClientSession", Any],
    params: ElicitRequestParams,
) -> ElicitResult:
    """Testing handler that automatically accepts with realistic test values.
    
    This handler is useful for integration tests where you want to verify
    the round-trip behavior of elicitation without user interaction.
    """
    logger.info(f"Auto-accept test handler called: {params.message}")
    
    if params.requestedSchema:
        # Generate realistic test data based on schema
        content = _generate_test_response(params.requestedSchema)
        return ElicitResult(action="accept", content=content)
    else:
        return ElicitResult(action="accept", content={"response": "auto-test-response"})


async def auto_decline_test_handler(
    context: RequestContext["ClientSession", Any],
    params: ElicitRequestParams,
) -> ElicitResult:
    """Testing handler that always declines elicitation requests."""
    logger.info(f"Auto-decline test handler called: {params.message}")
    return ElicitResult(action="decline")


async def auto_cancel_test_handler(
    context: RequestContext["ClientSession", Any],
    params: ElicitRequestParams,
) -> ElicitResult:
    """Testing handler that always cancels elicitation requests."""
    logger.info(f"Auto-cancel test handler called: {params.message}")
    return ElicitResult(action="cancel")


def _generate_test_response(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Generate realistic test data based on JSON schema."""
    if not schema or "properties" not in schema:
        return {"response": "default-test"}
    
    properties = schema["properties"]
    content = {}
    
    for field_name, field_def in properties.items():
        field_type = field_def.get("type", "string")
        
        if field_type == "string":
            # Provide field-specific test values
            if field_name.lower() in ["name", "full_name", "fullname"]:
                content[field_name] = "Test User"
            elif field_name.lower() in ["email", "email_address"]:
                content[field_name] = "test@example.com"
            elif field_name.lower() == "role":
                # Use enum values if available
                enum_values = field_def.get("enum", [])
                content[field_name] = enum_values[0] if enum_values else "developer"
            elif field_name.lower() in ["phone", "telephone", "phone_number"]:
                content[field_name] = "555-0123"
            elif field_name.lower() in ["address", "street_address"]:
                content[field_name] = "123 Test Street"
            elif field_name.lower() in ["city"]:
                content[field_name] = "Test City"
            elif field_name.lower() in ["country"]:
                content[field_name] = "Test Country"
            else:
                content[field_name] = f"test-{field_name.replace('_', '-')}"
                
        elif field_type == "integer":
            if field_name.lower() == "age":
                content[field_name] = 30
            elif field_name.lower() in ["year", "birth_year"]:
                content[field_name] = 1990
            elif field_name.lower() in ["count", "quantity", "amount"]:
                content[field_name] = 5
            else:
                content[field_name] = 42
                
        elif field_type == "number":
            if field_name.lower() in ["price", "cost", "salary"]:
                content[field_name] = 50000.00
            elif field_name.lower() in ["rating", "score"]:
                content[field_name] = 4.5
            else:
                content[field_name] = 3.14
                
        elif field_type == "boolean":
            # Provide reasonable defaults for common boolean fields
            if field_name.lower() in ["subscribe", "subscribe_newsletter", "newsletter"]:
                content[field_name] = True
            elif field_name.lower() in ["active", "enabled", "verified"]:
                content[field_name] = True
            elif field_name.lower() in ["disabled", "deleted", "archived"]:
                content[field_name] = False
            else:
                content[field_name] = True
                
        elif field_type == "array":
            default_array = field_def.get("default", ["test-item"])
            content[field_name] = default_array
            
        elif field_type == "object":
            default_object = field_def.get("default", {"test": "value"})
            content[field_name] = default_object
    
    return content