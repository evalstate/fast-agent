from fastmcp import FastMCP

app = FastMCP("Single Prompt Fixture")


@app.prompt(name="prompt", description="this is from the prompt file")
def prompt() -> str:
    return "this is from the prompt file"


if __name__ == "__main__":
    app.run(transport="stdio")
