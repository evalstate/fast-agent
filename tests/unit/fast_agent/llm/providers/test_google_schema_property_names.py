from copy import deepcopy

import pytest
from mcp import Tool

from fast_agent.llm.provider.google.google_converter import GoogleConverter


@pytest.mark.parametrize(
    "name", ["const", "additionalProperties", "exclusiveMinimum", "exclusiveMaximum"]
)
@pytest.mark.parametrize("nested", [False, True])
def test_google_tool_schema_preserves_properties_named_like_keywords(
    name: str, nested: bool
) -> None:
    arguments = {
        "type": "object",
        "properties": {name: {"type": "integer", "description": "An explicit tool argument"}},
        "required": [name],
        "additionalProperties": False,
    }
    schema = (
        {"type": "object", "properties": {"options": arguments}, "required": ["options"]}
        if nested
        else arguments
    )
    original = deepcopy(schema)
    tool = Tool(name="configure", input_schema=schema)

    converted = GoogleConverter().convert_to_google_tools([tool])

    declarations = converted[0].function_declarations
    assert declarations is not None
    parameters = declarations[0].parameters
    assert parameters is not None
    if nested:
        assert parameters.properties is not None
        parameters = parameters.properties["options"]
    assert parameters.properties is not None
    assert name in parameters.properties
    assert parameters.required == [name]
    assert parameters.properties[name].type is not None
    assert parameters.properties[name].description == "An explicit tool argument"
    assert schema == original
