"""
Manual test script for sampling with tools.

Run from this directory:
    uv run test_sampling_with_tools.py

If sampling with tools is working correctly, you should see the secret code
"WHISKEY-TANGO-FOXTROT-42" in the output. If you don't see it, there's an
error in the sampling-with-tools flow.
"""

import asyncio

import fast_agent as fa

fast_agent = fa.FastAgent("Sampling with Tools Test")

# The secret code that should appear if sampling with tools works
EXPECTED_SECRET = "WHISKEY-TANGO-FOXTROT-42"


@fast_agent.agent(
    name="tester",
    instruction="You are a helpful assistant that uses tools to complete tasks.",
    servers=["sampling_tools"],
)
async def main():
    async with fast_agent.run() as agent:
        print("\n" + "=" * 60)
        print("Testing Sampling with Tools - Secret Code Verification")
        print("=" * 60)

        # Test the fetch_secret tool - this verifies the full sampling-with-tools flow
        result = await agent.tester("Call the fetch_secret tool")

        print("\n" + "-" * 60)
        print("RESULT:")
        print("-" * 60)
        print(result)
        print("-" * 60)

        # Check if the secret code is in the result
        if EXPECTED_SECRET in result:
            print(f"\n✓ SUCCESS: Found secret code '{EXPECTED_SECRET}' in response!")
            print("  Sampling with tools is working correctly.")
        else:
            print(f"\n✗ FAILURE: Secret code '{EXPECTED_SECRET}' NOT found in response!")
            print("  There may be an issue with the sampling-with-tools implementation.")

        print("=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
