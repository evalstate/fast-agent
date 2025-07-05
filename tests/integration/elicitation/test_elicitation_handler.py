"""
Custom elicitation handler for integration testing.

This module provides a test elicitation handler that other tests can import
to verify custom handler functionality.
"""

from typing import TYPE_CHECKING, Any, Dict

from mcp.shared.context import RequestContext
from mcp.types import ElicitRequestParams, ElicitResult

from mcp_agent.logging.logger import get_logger

if TYPE_CHECKING:
    from mcp import ClientSession

logger = get_logger(__name__)


async def custom_elicitation_handler(
    context: RequestContext["ClientSession", Any],
    params: ElicitRequestParams,
) -> ElicitResult:
    """Test handler that returns predictable responses for integration testing."""
    logger.info(f"Test elicitation handler called with: {params.message}")
    
    if params.requestedSchema:
        # Generate test data based on the schema for round-trip verification
        properties = params.requestedSchema.get("properties", {})
        content: Dict[str, Any] = {}
        
        # Provide test values for each field
        for field_name, field_def in properties.items():
            field_type = field_def.get("type", "string")
            
            if field_type == "string":
                if field_name == "name":
                    content[field_name] = "Test User"
                elif field_name == "email":
                    content[field_name] = "test@example.com"
                elif field_name == "role":
                    # Check for enum values
                    enum_values = field_def.get("enum", [])
                    content[field_name] = enum_values[0] if enum_values else "developer"
                else:
                    content[field_name] = f"test-{field_name}"
            elif field_type == "integer":
                if field_name == "age":
                    content[field_name] = 30
                else:
                    content[field_name] = 42
            elif field_type == "number":
                content[field_name] = 3.14
            elif field_type == "boolean":
                if field_name == "subscribe_newsletter":
                    content[field_name] = True
                else:
                    content[field_name] = False
            elif field_type == "array":
                content[field_name] = ["test-item"]
            elif field_type == "object":
                content[field_name] = {"test": "value"}
        
        logger.info(f"Test handler returning: {content}")
        return ElicitResult(action="accept", content=content)
    else:
        # No schema, return simple response
        content = {"response": "test-response-no-schema"}
        logger.info(f"Test handler returning: {content}")
        return ElicitResult(action="accept", content=content)