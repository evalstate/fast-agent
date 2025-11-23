import pytest


@pytest.mark.integration
@pytest.mark.asyncio
async def test_chain_passthrough(fast_agent):  # CHAIN OF 3 BASIC AGENTS
    fast = fast_agent

    @fast.agent(
        "url_fetcher",
        instruction="Look at the articles on the page of the given url and summarize each of the articles.",
        model="passthrough",
    )
    @fast.agent(
        "summary_writer",
        instruction="""
        Write the given text to a file named summary.txt, and output which article topic is the most relevant to college students.
        """,
        model="passthrough",
    )
    @fast.agent(
        "google_sheets_writer",
        instruction="""
        Based on the given text, write some key points to research on the topic to a new google spreadsheet with a title "Research on <topic>".
        """,
        model="passthrough",
    )
    @fast.chain(
        name="topic_writer",
        sequence=["url_fetcher", "summary_writer", "google_sheets_writer"],
        cumulative=False,
    )
    @fast.chain(
        name="topic_writer_cumulative",
        sequence=["url_fetcher", "summary_writer", "google_sheets_writer"],
        cumulative=True,
    )
    async def chain_workflow():  # Renamed from main to avoid conflicts, and wrapped inside the test
        async with fast.run() as agent:
            input_url = "https://www.nytimes.com"
            result = await agent.topic_writer.send(input_url)
            assert result == input_url

            result = await agent.topic_writer_cumulative.send("X")
            # Expect tagged responses from all agents in cumulative output
            assert "<fastagent:request>X</fastagent:request>" in result
            assert "<fastagent:response agent='url_fetcher'>X</fastagent:response>" in result
            assert "<fastagent:response agent='summary_writer'>X\nX</fastagent:response>" in result
            assert (
                "<fastagent:response agent='google_sheets_writer'>X\nX\nX\nX</fastagent:response>"
                in result
            )

    await chain_workflow()  # Call the inner function
