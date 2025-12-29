import anyio
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.types import ListRootsResult, Root
from pydantic import FileUrl


async def list_roots_callback(context):
    # Return some example roots - change these to any paths you want to expose
    return ListRootsResult(
        roots=[
            Root(
                uri=FileUrl("file://foo/bar"),
                name="Home Directory",
            ),
            Root(
                uri=FileUrl("file:///tmp"),
                name="Temp Directory",
            ),
        ]
    )


async def main():
    # Start the server as a subprocess
    server_params = StdioServerParameters(
        command="uv",
        args=["run", "root_test_server.py"],
    )

    # Connect to the server via stdio
    async with stdio_client(server_params) as (read_stream, write_stream):
        # Create a client session
        async with ClientSession(read_stream, write_stream, list_roots_callback=list_roots_callback) as session:
            # Initialize the session
            await session.initialize()

            # Send initialized notification (required after initialize)
            # This is handled internally by initialize() in ClientSession

            # Call list_roots to get the roots from the server
            try:
                roots_result = await session.call_tool("show_roots", {})
                print(f"Received roots: {roots_result}")

                # Print each root for clarity
                # for root in roots_result.roots:
                #     print(f"Root: {root.uri}, Name: {root.name or 'unnamed'}")
            except Exception as e:
                print(f"Error listing roots: {e}")


# Run the async main function
if __name__ == "__main__":
    anyio.run(main)
