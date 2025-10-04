import os

from mcp.server.fastmcp import FastMCP

app = FastMCP(name="ENV-GET server")


@app.tool(description="Returns ENV variable specified")
def get_env_var(key: str) -> str:
    return os.environ.get(key, "")


if __name__ == "__main__":
    app.run()
