from fastmcp import Context, FastMCP
from fastmcp.server.dependencies import get_context
from fastmcp.utilities.types import Image
from mcp_types import SamplingMessage, TextContent

from fast_agent.mcp.helpers.content_helpers import get_text

# Create a FastMCP server
mcp = FastMCP(name="FastStoryAgent")


@mcp.resource("resource://fast-agent/short-story/{topic}")
async def generate_short_story(topic: str):
    prompt = f"Please write a short story on the topic of {topic}."

    # Make a sampling request to the client
    result = await get_context().session.create_message(  # ty: ignore[deprecated]
        max_tokens=1024,
        messages=[SamplingMessage(role="user", content=TextContent(type="text", text=prompt))],
    )

    return get_text(result.content) or ""


@mcp.tool()
async def sample_with_image(ctx: Context):
    result = await ctx.session.create_message(  # ty: ignore[deprecated]
        max_tokens=1024,
        messages=[
            SamplingMessage(
                role="user",
                content=TextContent(
                    type="text",
                    text="What is the username in this image?",
                ),
            ),
            SamplingMessage(role="user", content=Image(path="image.png").to_image_content()),
        ],
    )

    return get_text(result.content) or ""


# Run the server when this file is executed directly
if __name__ == "__main__":
    mcp.run()
