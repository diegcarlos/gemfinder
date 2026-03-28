import asyncio
from collections import defaultdict

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db
from app.models import ImageType, Product, ProductImage
from app.schemas import ImagesByType, ProductImageSavedResponse, SearchResponse, SearchResult
from app.services.embedding_service import EmbeddingService
from app.services.image_service import load_image_from_bytes, normalise_image, parse_filename, remove_background
from app.services.s3_service import S3Service

router = APIRouter()

ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/gif",
    "image/bmp",
    "image/tiff",
}


@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Buscar produtos similares por imagem",
    description=(
        "Envia uma imagem e recebe os produtos mais similares do catálogo. "
        "A comparação usa o embedding CLIP da imagem de fundo branco de cada produto. "
        "As URLs são presignadas e válidas por 1 hora."
    ),
)
async def search_by_image(
    file: UploadFile = File(..., description="Imagem de consulta (JPEG, PNG, WebP, etc.)"),
    db: AsyncSession = Depends(get_db),
) -> SearchResponse:
    """
    1. Valida e decodifica a imagem enviada.
    2. Gera embedding CLIP (thread pool, não bloqueia o event loop).
    3. Busca os top-K produtos mais similares via pgvector.
    4. Para cada produto, busca todas as imagens (clean/environment/person).
    5. Gera URLs presignadas para todas as imagens.
    6. Retorna os resultados agrupados por tipo de imagem.
    """
    # ── 1. Validar e ler a imagem ─────────────────────────────────────
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Tipo não suportado: {file.content_type}.",
        )

    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Arquivo enviado está vazio.")

    try:
        image = load_image_from_bytes(raw_bytes)
        image = normalise_image(image)

        # Remove o fundo antes de gerar o embedding para equalizar
        # o domínio entre fotos reais e imagens fundobranco do catálogo.
        if settings.REMOVE_BG_ON_SEARCH:
            image = await asyncio.to_thread(remove_background, image)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    # ── 2. Gerar embedding CLIP ───────────────────────────────────────
    svc = EmbeddingService.get_instance()
    embedding: list[float] = await svc.generate_embedding_async(image)

    # ── 3. Buscar produtos similares no banco ─────────────────────────
    similar_rows = await _search_similar(db, embedding, limit=settings.TOP_K_RESULTS)
    if not similar_rows:
        return SearchResponse(results=[], total=0)

    # ── 4. Buscar todas as imagens dos produtos encontrados ───────────
    product_ids = [row.id for row in similar_rows]
    img_result = await db.execute(
        select(ProductImage).where(ProductImage.product_id.in_(product_ids))
    )
    all_images = img_result.scalars().all()

    # Agrupa as chaves S3 por produto e tipo
    # { product_id: { "clean": [...], "environment": [...], "person": [...] } }
    images_map: dict[str, dict[str, list[str]]] = defaultdict(
        lambda: {"clean": [], "environment": [], "person": []}
    )
    for img in all_images:
        images_map[img.product_id][img.type.value].append(img.url)

    # ── 5. Gerar URLs presignadas e montar resposta ───────────────────
    s3 = S3Service()
    results: list[SearchResult] = []

    for row in similar_rows:
        pid = row.id
        img_keys = images_map[pid]

        # Presigna a imagem principal (main_image_url do produto)
        main_signed = await s3.presign_async(row.main_image_url)

        # Presigna todas as imagens por tipo em paralelo seria ideal,
        # mas para simplicidade fazemos sequencialmente
        signed_by_type: dict[str, list[str]] = {"clean": [], "environment": [], "person": []}
        for img_type, keys in img_keys.items():
            for key in keys:
                signed_by_type[img_type].append(await s3.presign_async(key))

        results.append(
            SearchResult(
                product_id=pid,
                name=row.name,
                similarity=float(row.similarity),
                main_image=main_signed,
                images=ImagesByType(**signed_by_type),
            )
        )

    return SearchResponse(results=results, total=len(results))


@router.post(
    "/products/image",
    response_model=ProductImageSavedResponse,
    status_code=201,
    summary="Salvar imagem de produto via API",
    description=(
        "Envia uma imagem para um produto específico. "
        "Se o tipo for 'clean' (fundo branco), gera o embedding CLIP e cria/atualiza o produto. "
        "Para outros tipos, apenas salva a imagem associada ao produto existente."
    ),
)
async def save_product_image(
    file: UploadFile = File(..., description="Imagem do produto"),
    product_id: str = Form(..., description="ID do produto (ex: 11666008)"),
    image_type: ImageType = Form(
        default=ImageType.clean,
        description="Tipo da imagem: clean | environment | person",
    ),
    db: AsyncSession = Depends(get_db),
) -> ProductImageSavedResponse:
    """
    1. Valida e decodifica a imagem.
    2. Faz upload para o R2 — armazena a chave S3.
    3. Se image_type == clean: gera embedding e cria/atualiza o produto.
    4. Cria o registro ProductImage.
    5. Retorna a imagem salva com URL presignada.
    """
    # ── 1. Validar content type ───────────────────────────────────────
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail=f"Tipo não suportado: {file.content_type}.")

    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Arquivo enviado está vazio.")

    try:
        image = load_image_from_bytes(raw_bytes)
        image = normalise_image(image)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    # ── 2. Upload para o R2 ───────────────────────────────────────────
    s3 = S3Service()
    filename = file.filename or f"{product_id}_{image_type.value}.jpg"
    try:
        s3_key = await s3.upload_image_async(raw_bytes, filename)
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    # ── 3. Se for imagem clean: gera embedding e cria/atualiza produto ─
    if image_type == ImageType.clean:
        svc = EmbeddingService.get_instance()
        embedding: list[float] = await svc.generate_embedding_async(image)

        existing = await db.get(Product, product_id)
        if existing:
            existing.main_image_url = s3_key
            existing.embedding = embedding
        else:
            db.add(Product(
                id=product_id,
                name=product_id,
                main_image_url=s3_key,
                embedding=embedding,
            ))
        await db.flush()
    else:
        # Para outros tipos, o produto precisa existir (ter imagem clean antes)
        existing = await db.get(Product, product_id)
        if not existing:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Produto '{product_id}' não encontrado. "
                    "Envie primeiro a imagem do tipo 'clean' para criar o produto."
                ),
            )

    # ── 4. Salva o registro ProductImage ─────────────────────────────
    product_image = ProductImage(
        product_id=product_id,
        type=image_type,
        url=s3_key,
    )
    db.add(product_image)
    await db.commit()
    await db.refresh(product_image)

    # ── 5. Retorna com URL presignada ─────────────────────────────────
    signed_url = await s3.presign_async(s3_key)

    return ProductImageSavedResponse(
        id=str(product_image.id),
        product_id=product_id,
        type=image_type.value,
        url=signed_url,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _search_similar(db: AsyncSession, embedding: list[float], limit: int):
    """
    Retorna os produtos mais próximos via distância cosseno do pgvector.

    Filtra resultados com similaridade < MIN_SIMILARITY (padrão: 0.5) para
    garantir que apenas correspondências relevantes sejam retornadas.
    Usa o operador <=> (distância cosseno): similaridade = 1 - distância.
    """
    stmt = text(
        """
        SELECT
            id,
            name,
            main_image_url,
            1 - (embedding <=> CAST(:embedding AS vector)) AS similarity
        FROM products
        WHERE 1 - (embedding <=> CAST(:embedding AS vector)) >= :min_similarity
        ORDER BY embedding <=> CAST(:embedding AS vector)
        LIMIT :limit
        """
    )
    embedding_str = "[" + ",".join(f"{v:.8f}" for v in embedding) + "]"
    result = await db.execute(
        stmt,
        {
            "embedding": embedding_str,
            "limit": limit,
            "min_similarity": settings.MIN_SIMILARITY,
        },
    )
    return result.fetchall()
