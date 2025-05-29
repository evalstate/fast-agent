import asyncio
import base64
import mimetypes
from pathlib import Path
from typing import List, Union

from mcp.types import ImageContent, TextContent

from mcp_agent.core.fastagent import FastAgent
from mcp_agent.core.prompt import Prompt
from mcp_agent.core.request_params import RequestParams

AGENT_NAME = "tensorzero_image_tester"
TENSORZERO_MODEL = "tensorzero.test_chat"
TEXT_PROMPT = (
    "Provide a description of the similarities and differences between these three images."
)
LOCAL_IMAGE_FILES = [
    Path("./demo_images/clam.jpg"),
    Path("./demo_images/shrimp.png"),
    Path("./demo_images/crab.png"),
]

MY_T0_SYSTEM_VARS = {
    "TEST_VARIABLE_1": "Roses are red",
    "TEST_VARIABLE_2": "Violets are blue",
    "TEST_VARIABLE_3": "Sugar is sweet",
    "TEST_VARIABLE_4": "Vibe code responsibly 👍",
}

fast = FastAgent("TensorZero Image Demo - Base64 Only")


@fast.agent(
    name=AGENT_NAME,
    model=TENSORZERO_MODEL,
    request_params=RequestParams(template_vars=MY_T0_SYSTEM_VARS),
)
async def main():
    content_parts: List[Union[TextContent, ImageContent]] = []
    content_parts.append(TextContent(type="text", text=TEXT_PROMPT))

    for file_path in LOCAL_IMAGE_FILES:
        mime_type, _ = mimetypes.guess_type(file_path)
        if not mime_type or not mime_type.startswith("image/"):
            ext = file_path.suffix.lower()
            if ext == ".jpg" or ext == ".jpeg":
                mime_type = "image/jpeg"
            elif ext == ".png":
                mime_type = "image/png"
        if mime_type is None:
            mime_type = "image/png"  # Default fallback if still None

        with open(file_path, "rb") as image_file:
            image_bytes = image_file.read()

        encoded_data = base64.b64encode(image_bytes).decode("utf-8")
        content_parts.append(ImageContent(type="image", mimeType=mime_type, data=encoded_data))

    message = Prompt.user(*content_parts)
    async with fast.run() as agent_app:
        agent = getattr(agent_app, AGENT_NAME)
        await agent.send(message)


if __name__ == "__main__":
    asyncio.run(main())  # type: ignore
