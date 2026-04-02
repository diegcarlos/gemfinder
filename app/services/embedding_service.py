import asyncio
from pathlib import Path

import open_clip
import torch
from PIL import Image

from app.config import settings


class EmbeddingService:
    """
    Singleton que encapsula o modelo CLIP ViT-B/32 como extrator de features.

    Pipeline de inferência:
        1. Pré-processamento espacial: crop foreground (RGBA-aware) + padding quadrado
           → garante que o objeto ocupa ≥ 70% do canvas independente da entrada
        2. Preprocess CLIP: resize 224px + normalização
        3. Forward pass (base ou fine-tunado)
        4. Normalização L2 → 512-dim para cosine similarity

    Entrada aceita RGB e RGBA:
        • RGB  → crop por threshold branco (catálogo fundobranco)
        • RGBA → crop por canal alpha do rembg (fotos reais após remove_background_rgba)
    """

    _instance: "EmbeddingService | None" = None

    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        self.model.eval()
        self.model.to(self.device)

        self._finetuned = False
        self._version   = "base"
        model_path = Path(settings.FINETUNED_MODEL_PATH)
        if model_path.exists():
            self._load_finetuned(model_path)

    def _load_finetuned(self, path: Path) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.model.visual.load_state_dict(checkpoint["visual_state_dict"])
        self.model.eval()
        self._finetuned = True
        self._version   = checkpoint.get("version", "v1")
        print(
            f"✓ Pesos fine-tunados carregados: {path} "
            f"(versão {self._version}, "
            f"época {checkpoint.get('epoch', '?')}, "
            f"top1 {checkpoint.get('best_top1', checkpoint.get('loss', '?'))})"
        )

    @classmethod
    def get_instance(cls) -> "EmbeddingService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        cls._instance = None

    def generate_embedding(self, image: Image.Image) -> list[float]:
        """
        Gera embedding normalizado L2 a partir de uma imagem PIL.

        Aceita RGB (catálogo) e RGBA (saída do remove_background_rgba para buscas).
        O pré-processamento aplica crop foreground + pad antes do CLIP preprocess,
        garantindo que o objeto esteja bem centrado e ocupe a maior parte do canvas.

        Returns:
            Lista de 512 floats normalizados L2.
        """
        from app.services.image_preprocessing_service import (
            crop_to_foreground,
            pad_to_square,
            rgba_to_white_background,
        )

        # Crop usando a melhor máscara disponível (alpha ou threshold branco)
        image = crop_to_foreground(image)

        # Converte RGBA → RGB com fundo branco após o crop preciso
        if image.mode == "RGBA":
            image = rgba_to_white_background(image)
        elif image.mode != "RGB":
            image = image.convert("RGB")

        image = pad_to_square(image)

        # CLIP preprocess: resize 224 + ToTensor + Normalize
        tensor = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.model.encode_image(tensor)
            features = features / features.norm(dim=-1, keepdim=True)

        return features.squeeze(0).cpu().tolist()

    async def generate_embedding_async(self, image: Image.Image) -> list[float]:
        """Wrapper assíncrono — executa inferência em thread pool."""
        return await asyncio.to_thread(self.generate_embedding, image)

    @property
    def embedding_dim(self) -> int:
        return 512

    @property
    def model_name(self) -> str:
        if self._finetuned:
            return f"CLIP ViT-B/32 fine-tuned ({self._version})"
        return "CLIP ViT-B/32 base"

    @property
    def model_version(self) -> str:
        return self._version
