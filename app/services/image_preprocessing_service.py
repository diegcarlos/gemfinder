"""
Pipeline de pré-processamento de imagens para joias.

Problema que resolve:
    • Fotos do usuário: objeto pequeno, fundo complexo, enquadramento inconsistente
    • Catálogo: objeto centralizado, fundo branco, zoom-in
    → Essa diferença de domínio degrada os embeddings CLIP

Pipeline (aplicado em treino, ingestão e busca):
    1. Remoção de fundo (rembg) → retorna RGBA com canal alpha preciso
    2. Crop ao foreground usando alpha channel (muito mais preciso que threshold de branco)
    3. Enforce fill ratio: objeto deve ocupar ≥ MIN_FILL_RATIO do canvas final
    4. Padding quadrado com fundo branco (preserva aspect ratio, centraliza)
    5. Resize 224×224 (apenas quando `resize=True` — embedding service deixa para o CLIP preprocess)

Garantia de consistência:
    • Catálogo (fundobranco): crop_to_foreground via threshold branco → pad → CLIP
    • Busca (foto real):       remove_background_rgba → crop_to_foreground via alpha → pad → CLIP
    • Treino:                  mesma sequência das imagens de catálogo
"""

import io

import numpy as np
from PIL import Image

# Threshold de branco para imagens sem canal alpha (catálogo fundobranco)
_WHITE_THRESHOLD = 235

# Padding relativo ao tamanho do objeto antes do pad quadrado
_BBOX_PADDING = 0.04

# Objeto deve ocupar ao menos esta fração da dimensão menor do canvas final
# Se abaixo deste valor, o pad é reduzido para forçar o objeto a ser maior
MIN_FILL_RATIO = 0.70


# ── Funções base ──────────────────────────────────────────────────────────────

def remove_background_rgba(image: Image.Image) -> Image.Image:
    """
    Remove o fundo e retorna RGBA com canal alpha representando o foreground.

    Diferente de `remove_background` (que converte para branco imediatamente),
    esta versão preserva o alpha para que `crop_to_foreground` possa usar
    a máscara precisa do U2Net ao invés do threshold de branco.

    Args:
        image: PIL Image de entrada (qualquer modo).

    Returns:
        PIL Image RGBA com fundo transparente.
    """
    from rembg import remove as rembg_remove  # lazy import

    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    result_bytes = rembg_remove(buf.getvalue())
    return Image.open(io.BytesIO(result_bytes)).convert("RGBA")


def crop_to_foreground(image: Image.Image, padding: float = _BBOX_PADDING) -> Image.Image:
    """
    Recorta ao bounding-box do objeto usando a melhor máscara disponível.

    Estratégia de detecção:
        • Modo RGBA → usa canal alpha (máscara precisa do rembg/U2Net)
        • Modo RGB  → usa threshold de branco (para imagens fundobranco)

    Em ambos os casos, adiciona padding relativo ao tamanho do objeto para
    evitar cortes abruptos nas bordas da joia.

    Args:
        image:   PIL Image (RGB ou RGBA).
        padding: Padding adicional relativo ao tamanho do bounding-box (padrão 4%).

    Returns:
        PIL Image no mesmo modo que a entrada, recortada.
    """
    arr = np.array(image)

    if image.mode == "RGBA":
        # Alpha > 10 → foreground detectado pelo rembg
        mask = arr[:, :, 3] > 10
    else:
        # Pixels onde pelo menos um canal é escuro o suficiente → não é fundo branco
        mask = ~np.all(arr[:, :, :3] > _WHITE_THRESHOLD, axis=2)

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not rows.any() or not cols.any():
        return image  # objeto não detectado — devolve original

    rmin = int(np.where(rows)[0][0])
    rmax = int(np.where(rows)[0][-1])
    cmin = int(np.where(cols)[0][0])
    cmax = int(np.where(cols)[0][-1])

    h, w = arr.shape[:2]
    obj_h = rmax - rmin
    obj_w = cmax - cmin
    pad_r = max(1, int(obj_h * padding))
    pad_c = max(1, int(obj_w * padding))

    rmin = max(0, rmin - pad_r)
    rmax = min(h, rmax + pad_r)
    cmin = max(0, cmin - pad_c)
    cmax = min(w, cmax + pad_c)

    return image.crop((cmin, rmin, cmax, rmax))


def rgba_to_white_background(image: Image.Image) -> Image.Image:
    """
    Composta imagem RGBA sobre canvas branco.

    Args:
        image: PIL Image RGBA.

    Returns:
        PIL Image RGB com fundo branco.
    """
    if image.mode != "RGBA":
        return image.convert("RGB")
    background = Image.new("RGB", image.size, (255, 255, 255))
    background.paste(image.convert("RGB"), mask=image.split()[3])
    return background


def pad_to_square(image: Image.Image, fill_ratio: float = MIN_FILL_RATIO) -> Image.Image:
    """
    Centraliza a imagem em um canvas quadrado com fundo branco.

    Garante que o objeto ocupe pelo menos `fill_ratio` da dimensão menor
    do canvas. Se o aspect ratio for muito extremo (objeto muito estreito),
    o padding é reduzido para forçar o objeto a ser visível.

    Args:
        image:      PIL Image RGB.
        fill_ratio: Fração mínima do canvas que o objeto deve ocupar (padrão 70%).

    Returns:
        PIL Image RGB quadrada.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    w, h = image.size
    size = max(w, h)

    # Verificação de fill ratio: se o objeto for estreito, reduz o padding extra
    min_dim = min(w, h)
    if min_dim / size < fill_ratio:
        # Recalcula o canvas para garantir o fill mínimo
        size = int(min_dim / fill_ratio)

    canvas = Image.new("RGB", (size, size), (255, 255, 255))
    paste_x = (size - w) // 2
    paste_y = (size - h) // 2
    canvas.paste(image, (paste_x, paste_y))
    return canvas


# ── Pipeline completo ─────────────────────────────────────────────────────────

def preprocess_for_embedding(
    image: Image.Image,
    remove_bg: bool = False,
    target_size: int = 224,
) -> Image.Image:
    """
    Pipeline completo de pré-processamento para geração de embeddings.

    Para imagens do catálogo (fundobranco):
        remove_bg=False → crop por threshold branco → pad → resize

    Para fotos do usuário (fundo real):
        remove_bg=True  → rembg RGBA → crop por alpha channel → branco → pad → resize

    O uso do canal alpha (quando disponível) dá uma máscara muito mais
    precisa do que o threshold de branco, garantindo que o objeto seja
    bem centralizado mesmo em fotos com iluminação variada.

    Args:
        image:       PIL Image de entrada (qualquer modo).
        remove_bg:   Se True, aplica rembg antes do crop.
        target_size: Dimensão final (padrão 224 para CLIP ViT-B/32).

    Returns:
        PIL Image RGB 224×224 pronta para o preprocess do CLIP.
    """
    if remove_bg:
        image = remove_background_rgba(image)       # → RGBA, alpha preciso
        image = crop_to_foreground(image)           # usa alpha channel
        image = rgba_to_white_background(image)     # → RGB, fundo branco
    else:
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = crop_to_foreground(image)           # usa threshold branco

    image = pad_to_square(image)
    image = image.resize((target_size, target_size), Image.LANCZOS)
    return image
