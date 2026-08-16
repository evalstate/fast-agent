from fastmcp import FastMCP
from fastmcp.prompts import Message

app = FastMCP("Playback Prompt Fixture")


@app.prompt(name="playback", description="[USER] user1 assistant1 user2")
def playback() -> list[Message]:
    return [
        Message("user1"),
        Message("assistant1", role="assistant"),
        Message("user2"),
    ]


if __name__ == "__main__":
    app.run(transport="stdio")
