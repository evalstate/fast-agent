from fastapi import APIRouter

router = APIRouter(
    prefix="/health",
    tags=["health"],
)


@router.get("/")
async def health_check():
    """
    Basic health check endpoint.
    """
    return {"status": "healthy", "message": "Service is running"}


@router.get("/info")
async def health_info():
    """
    Return basic information about the service.
    """
    return {
        "status": "healthy",
        "service_name": "BeeBackend",
        "version": "0.1.0",
    }