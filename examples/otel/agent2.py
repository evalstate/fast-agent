import asyncio
from typing import Annotated

from pydantic import BaseModel, Field

from fast_agent import FastAgent
from fast_agent.core.prompt import Prompt
from fast_agent.llm.request_params import RequestParams

# Create the application
fast = FastAgent("fast-agent example")


class FormattedResponse(BaseModel):
    thinking: Annotated[
        str, Field(description="Your reflection on the conversation that is not seen by the user.")
    ]
    message: str


# Define the agent
@fast.agent(
    name="chat",
    instruction="You are a helpful AI Agent",
    servers=["fetch"],
    request_params=RequestParams(maxTokens=8192),
)
async def main():
    # use the --model command line switch or agent arguments to change model
    async with fast.run() as agent:
        thinking, response = await agent.chat.generate(
            multipart_messages=[Prompt.user("Let's talk about guitars. Fetch from wikipedia")],
        )


if __name__ == "__main__":
    asyncio.run(main())
