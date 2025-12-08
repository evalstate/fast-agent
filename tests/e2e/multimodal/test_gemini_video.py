import base64
from pathlib import Path

import pytest
from mcp.types import BlobResourceContents, EmbeddedResource
from pydantic import AnyUrl

from fast_agent.types import PromptMessageExtended, video_link


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.parametrize(
    "model_name",
    [
        "gemini25",
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
                        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                        name="sample_video",
                        description="Big Buck Bunny trailer",
                    ),
                ],
            )
            message.add_text("What is this video about? Give a brief description.")

            # Send directly via generate
            response = await agent.default.generate([message])

            # The response should mention something about the video content
            response_text = response.all_text().lower()
            # Big Buck Bunny is an animated film about a rabbit
            assert any(term in response_text for term in ["rick", "astley", "icon", "never"]), (
                f"Expected video-related content in response: {response}"
            )

    await agent_function()


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.parametrize(
    "model_name",
    [
        "gemini25",
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
            assert any(term in response_text for term in ["rick", "astley", "icon", "never"]), (
                f"Expected video-related content in response: {response}"
            )

    await agent_function()


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.parametrize(
    "model_name",
    [
        "gemini25",
    ],
)
async def test_gemini_video_local_content(fast_agent, model_name):
    """Test Gemini can process a locally uploaded video file."""
    fast = fast_agent
    video_path = Path(__file__).parent / "tmp6vsgdcet.mp4"
    assert video_path.exists(), f"Local video file not found at {video_path}"

    # Encode the local video as a BlobResource so it is uploaded with the request
    video_b64 = base64.b64encode(video_path.read_bytes()).decode("ascii")
    video_resource = EmbeddedResource(
        type="resource",
        resource=BlobResourceContents(
            uri=AnyUrl(video_path.resolve().as_uri()),
            mimeType="video/mp4",
            blob=video_b64,
        ),
    )

    @fast.agent(
        "default",
        instruction="You analyze video content. Describe what you see.",
        model=model_name,
    )
    async def agent_function():
        async with fast.run() as agent:
            message = PromptMessageExtended(role="user", content=[video_resource])
            message.add_text("What is this video about? Give a brief description.")

            response = await agent.default.generate([message])
            response_text = response.all_text().lower()
            assert any(
                term in response_text for term in ["pet", "cat", "ginger", "feline", "window"]
            ), f"Expected video-related content in response: {response}"

    await agent_function()


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.parametrize(
    "model_name",
    [
        "gemini25",
    ],
)
async def test_gemini_image_resource_link_direct(fast_agent, model_name):
    pytest.skip("Image upload path pending files API support.")
