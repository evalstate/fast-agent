"""Provider-free MCP fixture recording actual tool execution."""

import sys
from pathlib import Path

from fastmcp import FastMCP

app = FastMCP("ptc-recording")
record = Path(sys.argv[1])


@app.tool()
def record_call(value: str) -> str:
    with record.open("a") as output:
        output.write(value + "\n")
    return value


if __name__ == "__main__":
    app.run(transport="stdio", show_banner=False)
