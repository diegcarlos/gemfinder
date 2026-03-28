from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.config import settings

# Create async engine — DATABASE_URL must use postgresql+asyncpg:// scheme
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=False,
    pool_pre_ping=True,  # Verifica saúde da conexão antes de reutilizar
    pool_size=10,
    max_overflow=20,
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    pass


async def get_db():
    """FastAPI dependency que fornece uma sessão assíncrona do banco."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_db() -> None:
    """
    Inicializa o banco de dados:
    - Habilita a extensão pgvector
    - Cria todas as tabelas (products + product_images)
    - Cria índice HNSW na coluna embedding para busca rápida por similaridade
    """
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

        # Importa os models para popular Base.metadata antes do create_all
        from app import models  # noqa: F401

        await conn.run_sync(Base.metadata.create_all)

        # Índice HNSW para similaridade cosseno (pgvector >= 0.5.0)
        # CLIP ViT-B/32 gera vetores de 512 dimensões — bem abaixo do limite de 2000
        await conn.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS ix_products_embedding_hnsw
                ON products
                USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64)
                """
            )
        )
