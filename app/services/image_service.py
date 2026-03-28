import io
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PIL import Image, UnidentifiedImageError

# Extensões suportadas para filtragem no import em lote
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tiff"}
)

# Tamanho máximo para evitar imagens enormes consumindo memória
MAX_IMAGE_SIZE: int = 4096

# Mapeamento: sufixo no nome do arquivo → ImageType
_TYPE_MAP: dict[str, str] = {
    "fundobranco": "clean",
    "ambiente": "environment",
    "pessoa": "person",
}


@dataclass
class ParsedFilename:
    """Resultado do parse de um nome de arquivo de produto."""

    product_id: str   # ex: "11666008"
    image_type: str   # "clean" | "environment" | "person"


def parse_filename(filename: str) -> Optional[ParsedFilename]:
    """
    Extrai product_id e image_type de um nome de arquivo no formato
    '{product_id}_{type}.ext'.

    Retorna None se o arquivo não corresponder a nenhum tipo conhecido
    (deve ser ignorado no pipeline de ingestão).

    Examples:
        "11666008_fundobranco.png" → ParsedFilename("11666008", "clean")
        "11666008_ambiente.png"   → ParsedFilename("11666008", "environment")
        "11666008_pessoa.png"     → ParsedFilename("11666008", "person")
        "thumbnail.jpg"           → None
    """
    stem = Path(filename).stem  # remove a extensão
    for raw_type, mapped_type in _TYPE_MAP.items():
        suffix = f"_{raw_type}"
        if stem.endswith(suffix):
            product_id = stem[: -len(suffix)]
            if product_id:  # garante que product_id não ficou vazio
                return ParsedFilename(product_id=product_id, image_type=mapped_type)
    return None


def load_image_from_bytes(data: bytes) -> Image.Image:
    """
    Decodifica bytes brutos em um objeto PIL Image.

    Raises:
        ValueError: Se os bytes não representarem uma imagem válida.
    """
    try:
        image = Image.open(io.BytesIO(data))
        image.load()  # força a decodificação para que erros apareçam aqui
        return image
    except UnidentifiedImageError as exc:
        raise ValueError(f"Formato de imagem não reconhecido: {exc}") from exc


def normalise_image(image: Image.Image, max_size: int = MAX_IMAGE_SIZE) -> Image.Image:
    """
    Redimensiona a imagem para que nenhuma dimensão exceda max_size,
    preservando o aspect ratio.
    """
    w, h = image.size
    if w <= max_size and h <= max_size:
        return image

    scale = max_size / max(w, h)
    return image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


def is_supported_image(filename: str) -> bool:
    """Retorna True se a extensão do arquivo estiver em SUPPORTED_EXTENSIONS."""
    return Path(filename).suffix.lower() in SUPPORTED_EXTENSIONS


def remove_background(image: Image.Image) -> Image.Image:
    """
    Remove o fundo da imagem usando rembg (modelo U2Net).

    Converte para fundo branco após remoção, igualando o domínio visual
    às imagens fundobranco do catálogo e melhorando significativamente
    a similaridade cosseno em fotos tiradas em ambientes reais.

    O modelo U2Net (~170 MB) é baixado automaticamente na primeira chamada
    e cacheado em ~/.u2net/. Chamadas subsequentes são rápidas.

    Args:
        image: PIL Image em qualquer modo.

    Returns:
        PIL Image RGB com fundo branco.
    """
    from rembg import remove as rembg_remove  # import lazy — não penaliza o startup

    # rembg opera sobre bytes PNG
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")

    result_bytes = rembg_remove(buf.getvalue())

    # Resultado é RGBA com fundo transparente → composta sobre branco
    result = Image.open(io.BytesIO(result_bytes)).convert("RGBA")
    background = Image.new("RGBA", result.size, (255, 255, 255, 255))
    background.paste(result, mask=result.split()[3])
    return background.convert("RGB")
