import asyncio

from mcp_agent.core.fastagent import FastAgent

# Create the application
fast = FastAgent("Social Media Manager")


@fast.agent(
    "url_fetcher",
    "Given a URL, provide a complete and comprehensive summary",
    servers=["fetch"],
)
@fast.agent(
    "post_author",
    """
    Write a 280 character social media post for any given text. 
    Respond only with the post, never use hashtags.
    """,
)
@fast.agent("translate_fr", "Translate the text to French.")
@fast.agent("translate_de", "Translate the text to German.")
@fast.agent(
    "review",
    """
    Cleanly format the original content and translations for  review by a Social Media manager.
    Highlight any cultural sensitivities.
    """,
    model="sonnet",
)
@fast.parallel(
    "translated_plan",
    fan_out=["translate_fr", "translate_de"],
)
@fast.agent(
    "human_review_and_post",
    """
- You can send a social media post by saving it to a file name 'post-<lang>.md'.
- NEVER POST TO SOCIAL MEDIA UNLESS THE HUMAN HAS REVIEWED AND APPROVED.

Present the Social Media report to the Human, and then provide direct actionable questions to assist
the Human in posting the content. 

You are being connected to a Human now, the first message you receive will be a
Social Media report ready to review with the Human.

""",
    human_input=True,
    servers=["filesystem"],
)
@fast.chain(
    "post_writer",
    sequence=[
        "url_fetcher",
        "post_author",
        "translated_plan",
        "human_review_and_post",
    ],
)
async def main() -> None:
    async with fast.run() as agent:
        # using chain workflow
        await agent.post_writer.prompt()


if __name__ == "__main__":
    asyncio.run(main())
