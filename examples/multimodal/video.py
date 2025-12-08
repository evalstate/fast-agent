"""Minimal test of ResourceLink with Gemini via agent.generate()"""

import asyncio

from fast_agent import FastAgent, text_content, video_link
from fast_agent.types import PromptMessageExtended

fast = FastAgent("Video Resource Test")


@fast.agent()
async def main():
    async with fast.run() as agent:
        message = PromptMessageExtended(
            role="user",
            content=[
                text_content("What happens in this video?."),
                video_link("https://www.youtube.com/watch?v=dQw4w9WgXcQ", name="Mystery Video"),
            ],
        )
        await agent.default.generate([message])


if __name__ == "__main__":
    asyncio.run(main())
