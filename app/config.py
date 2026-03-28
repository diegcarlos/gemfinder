from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Database
    DATABASE_URL: str

    # Cloudflare R2 (S3-compatible)
    R2_ACCESS_KEY_ID: str
    R2_SECRET_ACCESS_KEY: str
    R2_ACCOUNT_ID: str           # Found in R2 dashboard — used to build the endpoint URL
    R2_BUCKET_NAME: str
    R2_PUBLIC_URL: str           # Base public URL, e.g. https://pub-xxx.r2.dev or custom domain
    R2_REGION: str = "auto"      # R2 ignores region; "auto" is the conventional value

    # Segurança
    API_KEY: str  # Chave exigida em todas as requisições via header X-API-Key

    # App settings
    EMBEDDING_DIM: int = 512   # CLIP ViT-B/32 → 512 dimensões (bem abaixo do limite HNSW de 2000)
    TOP_K_RESULTS: int = 5
    MIN_SIMILARITY: float = 0.5   # Resultados abaixo deste score são descartados

    # Remove o fundo da imagem de consulta antes de gerar o embedding.
    # Equaliza o domínio entre fotos de celular e imagens fundobranco do catálogo.
    # Desative apenas para debug ou se as queries já chegarem sem fundo.
    REMOVE_BG_ON_SEARCH: bool = True

    # Fine-tuning
    # Caminho para os pesos fine-tunados. Se o arquivo existir, o EmbeddingService
    # os carrega automaticamente sobre o CLIP base. Deixe vazio para usar CLIP puro.
    FINETUNED_MODEL_PATH: str = "models/clip_jewelry.pt"

    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
