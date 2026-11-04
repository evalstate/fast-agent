"""Discover Dreamtune MCP tools or invoke one through the passthrough model."""

import argparse
import asyncio
import json
import os
from pathlib import Path

from fast_agent import FastAgent

fast = FastAgent(
    "Dreamtune MCP example",
    config_path=str(Path(__file__).with_name("fast-agent.yaml")),
    parse_cli_args=False,
    quiet=True,
)


@fast.agent(
    name="dreamtune",
    instruction="Invoke only the explicitly requested Dreamtune tool.",
    servers=["dreamtune"],
    model="passthrough",
)
async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tool", help="Tool name from the discovery output")
    parser.add_argument("--arguments", default="{}", help="JSON object for --tool")
    options = parser.parse_args()
    if not os.environ.get("DREAMTUNE_API_KEY"):
        parser.error("Set DREAMTUNE_API_KEY to your workspace API key")
    try:
        arguments = json.loads(options.arguments)
    except json.JSONDecodeError as error:
        parser.error(str(error))
    if not isinstance(arguments, dict):
        parser.error("--arguments must be a JSON object")

    async with fast.run() as agents:
        agent = agents["dreamtune"]
        tools = await agent.list_tools()
        if options.tool:
            if options.tool not in {tool.name for tool in tools.tools}:
                parser.error("Unknown tool; run without --tool to list available names")
            print(await agent.send(f"***CALL_TOOL {options.tool} {json.dumps(arguments)}"))
        else:
            for tool in tools.tools:
                print(f"{tool.name}: {tool.description}")


if __name__ == "__main__":
    asyncio.run(main())
