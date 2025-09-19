from contextlib import asynccontextmanager

from fastapi import Body, FastAPI, HTTPException

from fast_agent import PromptMessageExtended
from fast_agent.agents import McpAgent
from fast_agent.agents.agent_types import AgentConfig
from fast_agent.core import Core
from fast_agent.core.direct_factory import get_model_factory

core = Core(name="fast-agent core")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Manual lifecycle control: initialize Core and Agent explicitly
    await core.initialize()

    cfg = AgentConfig(
        name="core_agent",
        instruction="You are a helpful AI Agent.",
    )

    agent = McpAgent(config=cfg, context=core.context)
    await agent.initialize()

    llm_factory = get_model_factory(core.context, model=cfg.model)
    await agent.attach_llm(llm_factory)

    app.state.agent = agent
    try:
        yield
    finally:
        # Manual shutdown/cleanup
        try:
            await agent.shutdown()
        finally:
            await core.cleanup()


app = FastAPI(lifespan=lifespan)


@app.post("/ask", response_model=PromptMessageExtended)
async def ask(body: str = Body(..., media_type="text/plain")) -> PromptMessageExtended:
    try:
        # Call generate() to return the full multipart message (BaseModel)
        result = await app.state.agent.generate(body)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
