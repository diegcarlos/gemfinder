"""
Pipeline de pré-processamento de imagens para joias.

Etapas aplicadas antes de gerar embeddings:
    1. Conversão para RGB
    2. (Opcional) Remoção de fundo — rembg/U2Net
    3. Crop ao bounding-box do objeto (remove bordas brancas/vazias)
    4. Padding quadrado com fundo branco (preserva aspect ratio)
    5. Resize para 224×224 (input padrão do CLIP ViT-B/32)
    6. Normalização CLIP

O pipeline equaliza o domínio visual entre:
    • Fotos tiradas em celular (fundo real → remove_bg → fundo branco)
    • Imagens fundobranco do catálogo (já com fundo claro)

Uso:
    from app.services.image_preprocessing_service import preprocess_for_embedding
    processed = preprocess_for_embedding(pil_image, remove_bg=True)
"""

import numpy as np
from PIL import Image

# Limiar de branco: pixels com todos os canais acima deste valor são considerados fundo
_WHITE_THRESHOLD = 240

# Padding relativo ao tamanho do bounding-box do objeto (5% em cada lado)
_BBOX_PADDING = 0.05


def crop_to_object(image: Image.Image) -> Image.Image:
    """
    Recorta a imagem ao bounding-box do objeto (pixels não-brancos).

    Adiciona 5% de padding ao redor do objeto para evitar cortes abruptos.
    Se a imagem for completamente branca (objeto não detectado), retorna original.

    Args:
        image: PIL Image RGB.

    Returns:
        PIL Image RGB recortada.
    """
    arr = np.array(image.convert("RGB"))
    # Máscara: True onde o pixel NÃO é branco
    foreground = ~np.all(arr > _WHITE_THRESHOLD, axis=2)

    rows = np.any(foreground, axis=1)
    cols = np.any(foreground, axis=0)

    if not rows.any() or not cols.any():
        # Imagem completamente branca — devolve como está
        return image

    rmin, rmax = int(np.where(rows)[0][0]),  int(np.where(rows)[0][-1])
    cmin, cmax = int(np.where(cols)[0][0]),  int(np.where(cols)[0][-1])

    h, w = arr.shape[:2]
    pad_r = max(1, int((rmax - rmin) * _BBOX_PADDING))
    pad_c = max(1, int((cmax - cmin) * _BBOX_PADDING))

    rmin = max(0, rmin - pad_r)
    rmax = min(h, rmax + pad_r)
    cmin = max(0, cmin - pad_c)
    cmax = min(w, cmax + pad_c)

    return image.crop((cmin, rmin, cmax, rmax))


def pad_to_square(image: Image.Image) -> Image.Image:
    """
    Adiciona padding branco para tornar a imagem quadrada.

    Centraliza o objeto no canvas quadrado, preservando o aspect ratio.

    Args:
        image: PIL Image RGB.

    Returns:
        PIL Image RGB quadrada.
    """
    w, h = image.size
    size = max(w, h)
    canvas = Image.new("RGB", (size, size), (255, 255, 255))
    canvas.paste(image, ((size - w) // 2, (size - h) // 2))
    return canvas


def preprocess_for_embedding(
    image: Image.Image,
    remove_bg: bool = False,
    target_size: int = 224,
) -> Image.Image:
    """
    Pipeline completo de pré-processamento para geração de embeddings.

    Etapas:
        1. Converte para RGB
        2. (Opcional) Remove fundo com rembg
        3. Recorta ao bounding-box do objeto
        4. Centraliza em canvas quadrado com fundo branco
        5. Redimensiona para target_size × target_size

    O objeto PIL retornado ainda não está normalizado — a normalização
    CLIP é aplicada pelo preprocess do open_clip (ToTensor + Normalize).

    Args:
        image:       PIL Image de entrada.
        remove_bg:   Se True, aplica rembg antes do crop. Use para fotos de
                     celular; NÃO é necessário para imagens fundobranco.
        target_size: Dimensão do output (padrão 224 para ViT-B/32).

    Returns:
        PIL Image RGB pronta para ser processada pelo preprocess do CLIP.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    if remove_bg:
        from app.services.image_service import remove_background
        image = remove_background(image)

    image = crop_to_object(image)
    image = pad_to_square(image)
    image = image.resize((target_size, target_size), Image.LANCZOS)

    return image
