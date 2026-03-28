from fastapi import APIRouter

from app.schemas import HealthResponse
from app.services.embedding_service import EmbeddingService

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="API health check",
    description="Returns the current operational status of the API and its loaded model.",
)
async def health_check() -> HealthResponse:
    """
    Lightweight health probe used by load balancers and monitoring tools.

    Accessing the EmbeddingService singleton here also verifies that the model
    loaded successfully at startup.
    """
    svc = EmbeddingService.get_instance()
    return HealthResponse(
        status="ok",
        model=svc.model_name,
        embedding_dim=svc.embedding_dim,
    )
