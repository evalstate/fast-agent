from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import health
from .config import settings

app = FastAPI(
    title="BeeBackend API",
    description="Backend API for the BeeBackend project",
    version="0.1.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router)

@app.get("/")
async def root():
    return {"message": "Welcome to BeeBackend API"}