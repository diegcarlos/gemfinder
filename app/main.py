from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.database import init_db
from app.dependencies import verify_api_key
from app.routes import search, upload
from app.services.embedding_service import EmbeddingService


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.

    Startup:
        - Run database migrations / table creation (idempotent)
        - Pre-load the ResNet50 model into memory so the first request
          is not penalised by model loading time

    Shutdown:
        - Placeholder for graceful cleanup (connection pool teardown is
          handled automatically by SQLAlchemy's engine)
    """
    # ── Startup ──────────────────────────────────────────────────────
    await init_db()
    EmbeddingService.get_instance()  # warm up model
    print("✓ Database ready")
    print("✓ Embedding model loaded")

    yield  # application runs here

    # ── Shutdown ─────────────────────────────────────────────────────
    print("Shutting down GemFinder API…")


app = FastAPI(
    title="GemFinder — Jewelry Similarity Search",
    description=(
        "Upload a jewelry image and instantly find the most visually similar "
        "products in the catalogue using deep learning embeddings and pgvector."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Allow all origins in development; tighten for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rotas protegidas — exigem header X-API-Key válido
app.include_router(upload.router, tags=["Search"], dependencies=[Depends(verify_api_key)])

# /health é público — útil para healthcheck de load balancer / infra
app.include_router(search.router, tags=["Health"])
