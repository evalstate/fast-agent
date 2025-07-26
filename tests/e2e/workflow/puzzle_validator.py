import sys

from mcp.server.fastmcp import FastMCP

# Create the FastMCP server
app = FastMCP(name="Puzzle Validator")

secret_number = int(sys.argv[1]) if len(sys.argv) > 1 else 1234


@app.tool(
    name="guess",
    description="Validates a secret number. Returns 'passed' if correct, 'failed' otherwise.",
)
def guess(guess: int) -> str:
    if guess == secret_number:
        return "passed"
    else:
        return "failed"


if __name__ == "__main__":
    # Run the server using stdio transport
    app.run(transport="stdio")
