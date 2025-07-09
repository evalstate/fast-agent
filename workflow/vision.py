import asyncio
import base64

from mcp.types import TextContent, ImageContent
from mcp_agent import PromptMessageMultipart
from mcp_agent.core.fastagent import FastAgent

fast = FastAgent("Vision Example")

@fast.agent(
    "vision_analyst",
    instruction="Analyze images and describe what you see in detail.",
    model="ollama.qwen2.5vl:72b-q8_0",
)
async def main() -> None:
    async with fast.run() as agent:
        # Using with_resource for image analysis
        text_content = TextContent(text="Identify the image",type="text")
        with open("image.png", "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

        image_content = ImageContent(data=encoded_image, mimeType="image/png", type="image")
        prompt = PromptMessageMultipart(role="user",content=[text_content, image_content])

        result = await agent.vision_analyst.send(prompt)
        print(result)

if __name__ == "__main__":
    asyncio.run(main())