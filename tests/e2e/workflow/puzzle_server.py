import sys

from mcp.server.fastmcp import FastMCP

# Create the FastMCP server
app = FastMCP(name="Puzzle Server")

# Get secret number from command line or default to 42
secret_number = int(sys.argv[1]) if len(sys.argv) > 1 else 42


@app.tool(
    name="guess",
    description="Guess the secret number. Returns 'high', 'lower' or 'correct'",
)
def guess(guess: int) -> str:
    if guess < secret_number:
        return "higher"
    elif guess > secret_number:
        return "lower"
    else:
        return "correct"


if __name__ == "__main__":
    # Run the server using stdio transport
    app.run(transport="stdio")
