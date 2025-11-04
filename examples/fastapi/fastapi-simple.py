from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from fast_agent import FastAgent

# Create FastAgent without parsing CLI args (plays nice with uvicorn)
fast = FastAgent("fast-agent demo", parse_cli_args=False, quiet=True)


# Register a simple default agent via decorator
@fast.agent(name="helper", instruction="You are a helpful AI Agent.", default=True)
async def decorator():
    pass


# Keep FastAgent running for the app lifetime
@asynccontextmanager
async def lifespan(app: FastAPI):
    async with fast.run() as agents:
        app.state.agents = agents
        yield


app = FastAPI(lifespan=lifespan)


class AskRequest(BaseModel):
    message: str


class AskResponse(BaseModel):
    response: str


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest) -> AskResponse:
    try:
        result = await app.state.agents.send(req.message)
        return AskResponse(response=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
