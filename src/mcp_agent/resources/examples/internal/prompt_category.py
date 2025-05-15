from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.prompts.base import AssistantMessage, UserMessage

mcp = FastMCP("MCP Root Tester")


@mcp.prompt(name="category_prompt", description="set up the category protocol")
def category_prompt():
    return [
        UserMessage("Cat"),
        AssistantMessage("animal"),
        UserMessage("dog"),
        AssistantMessage("animal"),
        UserMessage("quartz"),
        AssistantMessage("mineral"),
        #        UserMessage("the sun"),
    ]


if __name__ == "__main__":
    mcp.run()
