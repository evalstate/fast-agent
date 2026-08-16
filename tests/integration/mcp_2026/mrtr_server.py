from typing import Annotated

from mcp.server.mcpserver import Elicit, MCPServer, Resolve
from pydantic import BaseModel


class Profile(BaseModel):
    name: str
    age: int


def ask_profile() -> Elicit[Profile]:
    return Elicit("Provide a profile", Profile)


server = MCPServer("modern-mrtr")


@server.tool()
def create_profile(profile: Annotated[Profile, Resolve(ask_profile)]) -> str:
    return f"{profile.name}:{profile.age}"


@server.resource("modern://status")
def status() -> str:
    return "modern-ok"


@server.prompt()
def hello(name: str = "world") -> str:
    return f"Hello, {name}"


if __name__ == "__main__":
    server.run(transport="stdio")
