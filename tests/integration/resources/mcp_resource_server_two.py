from fastmcp import FastMCP

app = FastMCP("Second Resource Fixture")


@app.resource("resource://fast-agent/r2file1.txt")
def first_resource() -> str:
    return "test 3"


@app.resource("resource://fast-agent/r2file2.txt")
def second_resource() -> str:
    return "test 4"


if __name__ == "__main__":
    app.run(transport="stdio")
