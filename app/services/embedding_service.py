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
        1. Pré-processamento: crop ao objeto + padding quadrado (centraliza a joia)
        2. Preprocess CLIP: resize 224px + normalização ImageNet/CLIP
        3. Forward pass pelo encoder visual (base ou fine-tunado)
        4. Normalização L2 → embedding 512-dim pronto para cosine similarity

    Carregamento:
        • CLIP ViT-B/32 base (pesos OpenAI) sempre carregado primeiro.
        • Se FINETUNED_MODEL_PATH existir, os pesos do encoder visual são
          sobrepostos automaticamente — troca entre base e fine-tuned apenas
          movendo ou removendo o arquivo.

    Versionamento:
        • model_version retorna a versão do checkpoint fine-tunado (ex: "v2-triplet")
          ou "base" se apenas o CLIP original estiver em uso.
    """

    _instance: "EmbeddingService | None" = None

    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ── 1. Carrega CLIP base ──────────────────────────────────────
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        self.model.eval()
        self.model.to(self.device)

        # ── 2. Sobrepõe pesos fine-tunados se disponíveis ─────────────
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

    # ------------------------------------------------------------------
    # Singleton
    # ------------------------------------------------------------------

    @classmethod
    def get_instance(cls) -> "EmbeddingService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        cls._instance = None

    # ------------------------------------------------------------------
    # Embedding generation
    # ------------------------------------------------------------------

    def generate_embedding(self, image: Image.Image) -> list[float]:
        """
        Gera embedding normalizado L2 a partir de uma imagem PIL.

        O pipeline aplica crop/pad para centralizar o objeto antes do preprocess
        do CLIP, melhorando a qualidade dos embeddings para imagens de joias.

        Args:
            image: PIL Image (qualquer modo/tamanho).

        Returns:
            Lista de 512 floats normalizados L2.
        """
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Pré-processamento: centraliza o objeto (crop + pad quadrado)
        from app.services.image_preprocessing_service import crop_to_object, pad_to_square
        image = crop_to_object(image)
        image = pad_to_square(image)

        # Preprocess CLIP: resize 224px + normalização
        tensor = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.model.encode_image(tensor)
            features = features / features.norm(dim=-1, keepdim=True)

        return features.squeeze(0).cpu().tolist()

    async def generate_embedding_async(self, image: Image.Image) -> list[float]:
        """Wrapper assíncrono — executa inferência em thread pool."""
        return await asyncio.to_thread(self.generate_embedding, image)

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

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
