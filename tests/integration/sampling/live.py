import asyncio

from fast_agent import FastAgent

# Create the application with specified model
fast = FastAgent("FastAgent Example")


# Define the agent
@fast.agent(servers=["sampling_test", "slow_sampling"])
async def main():
    # use the --model command line switch or agent arguments to change model
    async with fast.run() as agent:
        result = await agent.send(f'***CALL_TOOL sampling_test{SEP}sample {"to_sample": "123foo"}')
        print(f"RESULT: {result}")

        result = await agent.send(f"***CALL_TOOL slow_sampling{SEP}sample_parallel")
        print(f"RESULT: {result}")


if __name__ == "__main__":
    asyncio.run(main())
