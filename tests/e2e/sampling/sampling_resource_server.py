from mcp.server.fastmcp import Context, FastMCP, Image
from mcp.types import SamplingMessage, TextContent

# Create a FastMCP server
mcp = FastMCP(name="FastStoryAgent")


@mcp.resource("resource://fast-agent/short-story/{topic}")
async def generate_short_story(topic: str):
    prompt = f"Please write a short story on the topic of {topic}."

    # Make a sampling request to the client
    result = await mcp.get_context().session.create_message(
        max_tokens=1024,
        messages=[SamplingMessage(role="user", content=TextContent(type="text", text=prompt))],
    )

    return result.content.text


@mcp.tool()
async def sample_with_image(ctx: Context):
    result = await ctx.session.create_message(
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

    return result.content.text


# Run the server when this file is executed directly
if __name__ == "__main__":
    mcp.run()
