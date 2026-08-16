from fastmcp import FastMCP
from fastmcp.prompts import Message

app = FastMCP("Working Directory Prompt Fixture")


@app.prompt(name="multi")
def multi() -> list[Message]:
    return [
        Message("good morning"),
        Message("how may i help you?", role="assistant"),
    ]


if __name__ == "__main__":
    app.run(transport="stdio")
