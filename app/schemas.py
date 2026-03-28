from pydantic import BaseModel


class ImagesByType(BaseModel):
    """URLs presignadas das imagens de um produto, agrupadas por tipo."""

    clean: list[str] = []
    environment: list[str] = []
    person: list[str] = []


class SearchResult(BaseModel):
    """Um produto retornado pela busca por similaridade."""

    product_id: str
    name: str
    similarity: float        # 0.0 (dissimilar) → 1.0 (idêntico)
    main_image: str          # URL presignada da imagem clean (principal)
    images: ImagesByType     # Todas as imagens do produto por tipo


class SearchResponse(BaseModel):
    """Resposta do endpoint de busca por imagem."""

    results: list[SearchResult]
    total: int


class ProductImageSavedResponse(BaseModel):
    """Imagem de produto salva via endpoint de upload individual."""

    id: str
    product_id: str
    type: str
    url: str  # URL presignada válida por 1 hora


class HealthResponse(BaseModel):
    status: str
    model: str
    embedding_dim: int
