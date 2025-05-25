import asyncio

from mcp_agent.core.fastagent import FastAgent

# Create the application
fast = FastAgent("FastAgent Example")


# Define the agent
@fast.agent(name="haiku", model="haiku")
@fast.agent(name="openai", model="o3-mini.medium")

# @fast.agent(name="test")
async def main() -> None:
    async with fast.run() as agent:
        # Start an interactive session with "haiku"
        await agent.prompt(agent_name="haiku")
        # Transfer the message history top "openai"
        await agent.openai.generate(agent.haiku.message_history)
        # Continue the conversation
        await agent.prompt(agent_name="openai")  # Interactive shell

        # result: str = await agent.send("foo")
        # mcp_prompt: PromptMessage = PromptMessage(
        #     role="user", content=TextContent(type="text", text="How are you?")
        # )
        # result: str = agent.send(mcp_prompt)
        # resource: ReadResourceResult = agent.openai.get_resource(
        #     "server_name", "resource://images/cat.png"
        # )
        # response: str = Prompt.user("What is in this image?", resource)


if __name__ == "__main__":
    asyncio.run(main())
