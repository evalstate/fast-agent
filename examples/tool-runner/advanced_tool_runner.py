"""
Advanced ToolRunner example showing customization features.

This example demonstrates:
- Inspecting tool responses before they're sent back
- Modifying request parameters between iterations
- Appending additional messages/guidance during the loop
"""

import asyncio

from mcp.types import TextContent

from fast_agent import FastAgent, PromptMessageExtended, RequestParams

fast = FastAgent("Advanced Tool Runner Example")


def search_database(query: str) -> str:
    """Search the database for information.

    Args:
        query: Search query

    Returns:
        Search results
    """
    # Simulated database
    data = {
        "revenue": "Q4 2024 revenue was $1.2M, up 15% YoY",
        "users": "Active users: 50,000 (up 20% from last quarter)",
        "products": "Top products: Widget Pro (40%), Widget Basic (35%), Widget Enterprise (25%)",
    }

    for key, value in data.items():
        if key in query.lower():
            return value

    return f"No results found for: {query}"


def send_email(to: str, subject: str, body: str) -> str:
    """Send an email.

    Args:
        to: Recipient email address
        subject: Email subject
        body: Email body

    Returns:
        Confirmation message
    """
    print(f"  [EMAIL] To: {to}")
    print(f"  [EMAIL] Subject: {subject}")
    print(f"  [EMAIL] Body: {body[:100]}...")
    return f"Email sent successfully to {to}"


@fast.agent(
    name="analyst",
    instruction="You are a business analyst. Use tools to gather data and send reports.",
    tools=[search_database, send_email],
)
async def main():
    async with fast.run() as app:
        print("=" * 70)
        print("Advanced ToolRunner: Inspect and modify during iteration")
        print("=" * 70)

        runner = app.tool_runner(
            "Search for our revenue data, then email a summary to boss@company.com"
        )

        async for message in runner:
            print(f"\n[Iteration {runner.iterations}] Stop reason: {message.stop_reason}")

            # Inspect tool calls
            if message.tool_calls:
                for call_id, call in message.tool_calls.items():
                    print(f"  Tool: {call.params.name}")
                    print(f"  Args: {call.params.arguments}")

            # Get the tool response before it's automatically sent
            tool_response = await runner.generate_tool_call_response()
            if tool_response and tool_response.tool_results:
                print("  Tool results preview:")
                for result_id, result in tool_response.tool_results.items():
                    for content in result.content:
                        if hasattr(content, "text"):
                            print(f"    -> {content.text[:80]}...")

            # Example: Add guidance after the first tool call
            if runner.iterations == 1 and not runner.is_done:
                print("\n  [Injecting guidance: 'Keep it brief']")
                runner.append_messages(
                    PromptMessageExtended(
                        role="user",
                        content=[
                            TextContent(
                                type="text",
                                text="Please keep the email summary brief - 2-3 sentences max.",
                            )
                        ],
                    )
                )

            # Example: Increase max tokens if we're doing complex work
            if runner.iterations == 2 and not runner.is_done:
                print("\n  [Adjusting params: increasing maxTokens]")
                runner.set_request_params(
                    lambda p: RequestParams(
                        **(p.model_dump() if p else {}),
                        maxTokens=2048,
                    )
                )

        print("\n" + "=" * 70)
        print(f"Completed in {runner.iterations} iterations")
        print("=" * 70)

        # Show final response
        if runner.current_message:
            final_text = runner.current_message.last_text()
            if final_text:
                print(f"\nFinal response:\n{final_text}")


if __name__ == "__main__":
    asyncio.run(main())
