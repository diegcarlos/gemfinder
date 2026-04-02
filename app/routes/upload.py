import asyncio
import io
from collections import defaultdict

import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from PIL import Image
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db
from app.models import ImageType, Product, ProductImage
from app.schemas import ImagesByType, ProductImageSavedResponse, SearchResponse, SearchResult
from app.services.embedding_service import EmbeddingService
from app.services.image_service import load_image_from_bytes, normalise_image, parse_filename
from app.services.image_preprocessing_service import remove_background_rgba
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
    2. Remove fundo (rembg) retornando RGBA para crop preciso pelo canal alpha.
    3. Gera embedding CLIP com pré-processamento consistente com o catálogo.
    4. Busca RERANK_CANDIDATES produtos via pgvector.
    5. Re-ranking: combina cosine similarity com histograma de cor HSV.
    6. Retorna TOP_K_RESULTS com URLs presignadas para todas as imagens.
    """
    # ── 1. Validar e ler imagem ───────────────────────────────────────
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

    # ── 2. Remoção de fundo → RGBA para crop preciso pelo alpha ──────
    query_image_rgb: Image.Image  # mantido em RGB para re-ranking por cor

    if settings.REMOVE_BG_ON_SEARCH:
        # remove_background_rgba preserva o canal alpha do U2Net
        # → crop_to_foreground no embedding_service usa alpha (mais preciso que threshold branco)
        rgba_image = await asyncio.to_thread(remove_background_rgba, image)
        query_image_rgb = await asyncio.to_thread(_rgba_to_rgb, rgba_image)
        embedding_input = rgba_image  # RGBA → embedding_service usa alpha para crop
    else:
        query_image_rgb = image.convert("RGB")
        embedding_input = query_image_rgb

    # ── 3. Gerar embedding CLIP ───────────────────────────────────────
    svc = EmbeddingService.get_instance()
    embedding: list[float] = await svc.generate_embedding_async(embedding_input)

    # ── 4. Buscar candidatos no pgvector ─────────────────────────────
    candidates_limit = (
        max(settings.RERANK_CANDIDATES, settings.TOP_K_RESULTS)
        if settings.RERANK_RESULTS
        else settings.TOP_K_RESULTS
    )
    similar_rows = await _search_similar(db, embedding, limit=candidates_limit)
    if not similar_rows:
        return SearchResponse(results=[], total=0)

    # ── 5. Re-ranking por histograma de cor HSV ───────────────────────
    if settings.RERANK_RESULTS and len(similar_rows) > 1:
        s3_for_rerank = S3Service()
        similar_rows = await _rerank_by_color(
            query_image_rgb, similar_rows, s3_for_rerank, settings.TOP_K_RESULTS
        )
    else:
        similar_rows = list(similar_rows)[: settings.TOP_K_RESULTS]

    # ── 6. Buscar imagens dos produtos e presignar URLs ───────────────
    product_ids = [row.id for row in similar_rows]
    img_result = await db.execute(
        select(ProductImage).where(ProductImage.product_id.in_(product_ids))
    )
    all_images = img_result.scalars().all()

    images_map: dict[str, dict[str, list[str]]] = defaultdict(
        lambda: {"clean": [], "environment": [], "person": []}
    )
    for img in all_images:
        images_map[img.product_id][img.type.value].append(img.url)

    s3 = S3Service()
    results: list[SearchResult] = []

    for row in similar_rows:
        pid = row.id
        img_keys = images_map[pid]
        main_signed = await s3.presign_async(row.main_image_url)

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

    s3 = S3Service()
    filename = file.filename or f"{product_id}_{image_type.value}.jpg"
    try:
        s3_key = await s3.upload_image_async(raw_bytes, filename)
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc))

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
        existing = await db.get(Product, product_id)
        if not existing:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Produto '{product_id}' não encontrado. "
                    "Envie primeiro a imagem do tipo 'clean' para criar o produto."
                ),
            )

    product_image = ProductImage(product_id=product_id, type=image_type, url=s3_key)
    db.add(product_image)
    await db.commit()
    await db.refresh(product_image)

    signed_url = await s3.presign_async(s3_key)
    return ProductImageSavedResponse(
        id=str(product_image.id),
        product_id=product_id,
        type=image_type.value,
        url=signed_url,
    )


# ── Helpers de busca ──────────────────────────────────────────────────────────

async def _search_similar(db: AsyncSession, embedding: list[float], limit: int):
    """
    Retorna os produtos mais próximos via distância cosseno do pgvector.
    Filtra resultados abaixo de MIN_SIMILARITY.
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
            "embedding":     embedding_str,
            "limit":         limit,
            "min_similarity": settings.MIN_SIMILARITY,
        },
    )
    return result.fetchall()


# ── Re-ranking por histograma de cor ─────────────────────────────────────────

async def _rerank_by_color(
    query_rgb: Image.Image,
    candidates: list,
    s3: S3Service,
    top_k: int,
    cosine_weight: float = 0.65,
    color_weight:  float = 0.35,
) -> list:
    """
    Re-ordena os candidatos combinando cosine similarity com histograma de cor HSV.

    Estratégia:
        • Cosine similarity (pgvector): captura forma, textura e semântica
        • Histograma HSV (H + S, ignora V): captura cor independente de iluminação
        • Score combinado = cosine_weight × cosine + color_weight × color_hist_sim

    Os dois canais se complementam: cosine detecta categoria correta,
    histograma de cor elimina falsos positivos com tonalidades diferentes
    (ex: anel dourado vs. anel prateado).

    Args:
        query_rgb:     Imagem de query em RGB (sem fundo).
        candidates:    Rows do pgvector (id, name, main_image_url, similarity).
        s3:            Serviço S3 para download das imagens do catálogo.
        top_k:         Número de resultados a retornar após re-ranking.
        cosine_weight: Peso da similaridade cosseno (padrão 0.65).
        color_weight:  Peso do histograma de cor (padrão 0.35).

    Returns:
        Lista de rows re-ordenada, com top_k elementos.
    """
    query_hist = _compute_hsv_histogram(query_rgb)

    # Baixa as imagens principais de todos os candidatos em paralelo
    async def download_hist(row) -> np.ndarray | None:
        try:
            img_bytes = await s3.download_image_async(row.main_image_url)
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            return _compute_hsv_histogram(img)
        except Exception:
            return None  # se falhar, usa apenas cosine

    hists = await asyncio.gather(*[download_hist(row) for row in candidates])

    # Calcula score combinado e re-ordena
    scored: list[tuple[any, float]] = []
    for row, hist in zip(candidates, hists):
        cosine = float(row.similarity)
        if hist is not None:
            color_sim = _histogram_intersection(query_hist, hist)
            combined  = cosine_weight * cosine + color_weight * color_sim
        else:
            combined = cosine
        scored.append((row, combined))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [row for row, _ in scored[:top_k]]


def _compute_hsv_histogram(
    image: Image.Image,
    h_bins: int = 32,
    s_bins: int = 16,
    resize_to: int = 64,
) -> np.ndarray:
    """
    Calcula histograma 2D dos canais H (matiz) e S (saturação) no espaço HSV.

    Ignora o canal V (valor/brilho) para ser invariante à iluminação.
    Redimensiona para 64×64 antes do cálculo para uniformizar e ganhar velocidade.

    Args:
        image:    PIL Image RGB.
        h_bins:   Bins para o canal Hue (padrão 32).
        s_bins:   Bins para o canal Saturation (padrão 16).
        resize_to: Tamanho de redimensionamento antes do cálculo (padrão 64).

    Returns:
        Array 1D normalizado de tamanho h_bins × s_bins.
    """
    img = image.resize((resize_to, resize_to), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0

    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    max_c = np.maximum(np.maximum(r, g), b)
    min_c = np.minimum(np.minimum(r, g), b)
    delta = max_c - min_c

    # Saturação S
    s = np.where(max_c > 1e-6, delta / max_c, 0.0)

    # Matiz H (normalizado para [0, 1])
    h = np.zeros_like(r)
    eps = 1e-6
    m_r = (max_c == r) & (delta > eps)
    m_g = (max_c == g) & (delta > eps)
    m_b = (max_c == b) & (delta > eps)
    h[m_r] = ((g[m_r] - b[m_r]) / delta[m_r]) % 6.0
    h[m_g] = (b[m_g] - r[m_g]) / delta[m_g] + 2.0
    h[m_b] = (r[m_b] - g[m_b]) / delta[m_b] + 4.0
    h = h / 6.0

    # Histograma 2D H×S
    hist, _ = np.histogramdd(
        np.stack([h.ravel(), s.ravel()], axis=1),
        bins=[h_bins, s_bins],
        range=[[0, 1], [0, 1]],
    )
    hist = hist.ravel().astype(np.float32)

    # Normaliza para soma = 1
    total = hist.sum()
    if total > 0:
        hist /= total

    return hist


def _histogram_intersection(h1: np.ndarray, h2: np.ndarray) -> float:
    """
    Similaridade por interseção de histogramas: sum(min(h1, h2)).

    Retorna valor em [0, 1]. Dois histogramas idênticos retornam 1.0.
    Invariante a deslocamentos de iluminação (graças ao canal H+S sem V).
    """
    return float(np.minimum(h1, h2).sum())


def _rgba_to_rgb(image: Image.Image) -> Image.Image:
    """Composta RGBA sobre fundo branco → RGB."""
    if image.mode != "RGBA":
        return image.convert("RGB")
    bg = Image.new("RGB", image.size, (255, 255, 255))
    bg.paste(image.convert("RGB"), mask=image.split()[3])
    return bg
