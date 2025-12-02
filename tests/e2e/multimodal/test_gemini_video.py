"""
E2E tests for Gemini video support via ResourceLink.

These tests verify that:
1. ResourceLink with video MIME types passes through content filtering
2. GoogleConverter correctly converts ResourceLink to Part.from_uri()
3. Gemini can fetch and process video content from URLs
"""

import pytest

from fast_agent.types import PromptMessageExtended, image_link, video_link


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.parametrize(
    "model_name",
    [
        "gemini-2.0-flash",
    ],
)
async def test_gemini_video_resource_link_direct(fast_agent, model_name):
    """Test that Gemini can process a video ResourceLink sent directly via generate()."""
    fast = fast_agent

    @fast.agent(
        "default",
        instruction="You analyze video content. Describe what you see.",
        model=model_name,
    )
    async def agent_function():
        async with fast.run() as agent:
            # Create a message with a video ResourceLink
            message = PromptMessageExtended(
                role="user",
                content=[
                    video_link(
                        "https://www.youtube.com/watch?v=aqz-KE-bpKQ",
                        name="sample_video",
                        description="Big Buck Bunny trailer",
                    ),
                ],
            )
            message.add_text("What is this video about? Give a brief description.")

            # Send directly via generate
            response = await agent.generate([message])

            # The response should mention something about the video content
            response_text = response.lower()
            # Big Buck Bunny is an animated film about a rabbit
            assert any(
                term in response_text
                for term in ["bunny", "rabbit", "animated", "animation", "cartoon", "character"]
            ), f"Expected video-related content in response: {response}"

    await agent_function()


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.parametrize(
    "model_name",
    [
        "gemini-2.0-flash",
    ],
)
async def test_gemini_video_resource_link_via_tool(fast_agent, model_name):
    """Test that Gemini can process a video ResourceLink returned by an MCP tool."""
    fast = fast_agent

    @fast.agent(
        "default",
        instruction="You analyze video content. When asked, use tools to get video links and describe what you see.",
        servers=["video_server"],
        model=model_name,
    )
    async def agent_function():
        async with fast.run() as agent:
            response = await agent.send(
                "Use the get_video_link tool to get a video, then describe what the video is about."
            )

            # The response should mention something about the video content
            response_text = response.lower()
            assert any(
                term in response_text
                for term in ["bunny", "rabbit", "animated", "animation", "cartoon", "character"]
            ), f"Expected video-related content in response: {response}"

    await agent_function()


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.parametrize(
    "model_name",
    [
        "gemini-2.0-flash",
    ],
)
async def test_gemini_image_resource_link_direct(fast_agent, model_name):
    """Test that Gemini can process an image ResourceLink."""
    fast = fast_agent

    @fast.agent(
        "default",
        instruction="You analyze images. Describe what you see.",
        model=model_name,
    )
    async def agent_function():
        async with fast.run() as agent:
            # Create a message with an image ResourceLink
            message = PromptMessageExtended(
                role="user",
                content=[
                    image_link(
                        "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/300px-PNG_transparency_demonstration_1.png",
                        name="sample_image",
                        description="PNG transparency demonstration",
                    ),
                ],
            )
            message.add_text("What do you see in this image?")

            response = await agent.generate([message])

            # The response should mention something about the image
            response_text = response.lower()
            assert any(
                term in response_text
                for term in ["dice", "transparent", "cube", "red", "image", "png"]
            ), f"Expected image-related content in response: {response}"

    await agent_function()
