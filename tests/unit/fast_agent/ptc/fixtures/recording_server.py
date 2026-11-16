"""Provider-free MCP fixture recording actual tool execution."""

import sys
from pathlib import Path

from fastmcp import FastMCP

app = FastMCP("ptc-recording", instructions="Records calls.")
record = Path(sys.argv[1])


@app.tool()
def record_call(value: str) -> str:
    with record.open("a") as output:
        output.write(value + "\n")
    return value


@app.tool(meta={"fast_agent": {"ptc": {"expose": "both"}}})
def allowed_call(value: str) -> str:
    return record_call(value)


@app.tool(meta={"fast_agent": {"ptc": {"expose": "both"}}})
def hidden_call(value: str) -> str:
    return record_call(value)


@app.tool(meta={"fast_agent": {"ptc": {"expose": "both"}}})
def read_text_file(path: str) -> str:
    return record_call(path)


@app.resource("recording://allowed/doc")
def allowed_doc() -> str:
    return record_call("allowed/doc")


@app.resource("recording://hidden/doc")
def hidden_doc() -> str:
    return record_call("hidden/doc")


@app.resource("recording://allowed/item/{item}")
def allowed_item(item: str) -> str:
    return record_call(f"allowed/item/{item}")


@app.resource("https://example.test/allowed/doc")
def https_allowed_doc() -> str:
    return record_call("https/allowed/doc")


@app.resource("https://example.test/hidden/doc")
def https_hidden_doc() -> str:
    return record_call("https/hidden/doc")


@app.resource("recording://host/hidden/doc")
def host_hidden_doc() -> str:
    return record_call("host/hidden/doc")


if __name__ == "__main__":
    app.run(transport="stdio", show_banner=False)
