from fastmcp import FastMCP

app = FastMCP("First Resource Fixture")


@app.resource("resource://fast-agent/r1file1.txt")
def first_resource() -> str:
    return "test 1"


@app.resource("resource://fast-agent/r1file2.txt")
def second_resource() -> str:
    return "test 2"


if __name__ == "__main__":
    app.run(transport="stdio")
