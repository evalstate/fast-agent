from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.prompts.base import AssistantMessage, UserMessage
from pydantic import Field

mcp = FastMCP("MCP Prompt Tester")


@mcp.prompt(name="sizing_prompt", description="set up the sizing protocol")
def sizing_prompt():
    return [
        UserMessage("What is the size of the moon?"),
        AssistantMessage("OBJECT: MOON\nSIZE: 3,474.8\nUNITS: KM\nTYPE: MINERAL"),
        UserMessage("What is the size of the Earth?"),
        AssistantMessage("OBJECT: EARTH\nSIZE: 12,742\nUNITS: KM\nTYPE: MINERAL"),
        UserMessage("A tiger"),
        AssistantMessage("OBJECT: TIGER\nSIZE: 1.2\nUNITS: M\nTYPE: ANIMAL"),
        UserMessage("Domestic Cat"),
    ]


@mcp.prompt(
    name="sizing_prompt_units",
    description="set up the sizing protocol with metric or imperial units",
)
def sizing_prompt_units(
    metric: bool = Field(description="Set to True for Metric, False for Imperial", default=True),
):
    if metric:
        return [
            UserMessage("What is the size of the moon?"),
            AssistantMessage("OBJECT: MOON\nSIZE: 3,474.8\nUNITS: KM\nTYPE: MINERAL"),
            UserMessage("What is the size of the Earth?"),
            AssistantMessage("OBJECT: EARTH\nSIZE: 12,742\nUNITS: KM\nTYPE: MINERAL"),
            UserMessage("A tiger"),
            AssistantMessage("OBJECT: TIGER\nSIZE: 1.2\nUNITS: M\nTYPE: ANIMAL"),
            UserMessage("Domestic Cat"),
        ]
    else:
        return [
            UserMessage("What is the size of the moon?"),
            AssistantMessage("OBJECT: MOON\nSIZE: 2,159.1\nUNITS: MI\nTYPE: MINERAL"),
            UserMessage("What is the size of the Earth?"),
            AssistantMessage("OBJECT: EARTH\nSIZE: 7,918\nUNITS: MI\nTYPE: MINERAL"),
            UserMessage("A tiger"),
            AssistantMessage("OBJECT: TIGER\nSIZE: 3.9\nUNITS: FT\nTYPE: ANIMAL"),
            UserMessage("Domestic Cat"),
        ]


if __name__ == "__main__":
    mcp.run()
