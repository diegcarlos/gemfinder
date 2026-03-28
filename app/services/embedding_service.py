import asyncio
from pathlib import Path

import open_clip
import torch
from PIL import Image

from app.config import settings


class EmbeddingService:
    """
    Singleton que encapsula o modelo CLIP ViT-B/32 como extrator de features.

    Comportamento:
      1. Carrega o CLIP ViT-B/32 base com pesos do OpenAI.
      2. Se FINETUNED_MODEL_PATH apontar para um arquivo existente, os pesos
         fine-tunados do encoder visual são sobrepostos automaticamente.
         Isso permite trocar entre base e fine-tuned apenas movendo/removendo o arquivo.

    Vetores de saída: 512-dim, normalizados L2.
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
        model_path = Path(settings.FINETUNED_MODEL_PATH)
        if model_path.exists():
            self._load_finetuned(model_path)

    def _load_finetuned(self, path: Path) -> None:
        """
        Carrega os pesos do encoder visual fine-tunado sobre o CLIP base.

        O checkpoint contém apenas o state_dict de model.visual — não o modelo
        completo — para manter o arquivo pequeno e o carregamento rápido.
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.model.visual.load_state_dict(checkpoint["visual_state_dict"])
        self.model.eval()
        self._finetuned = True
        print(
            f"✓ Pesos fine-tunados carregados: {path} "
            f"(época {checkpoint.get('epoch', '?')}, "
            f"loss {checkpoint.get('loss', '?'):.4f})"
        )

    # ------------------------------------------------------------------
    # Singleton accessor
    # ------------------------------------------------------------------

    @classmethod
    def get_instance(cls) -> "EmbeddingService":
        """Retorna (ou cria lazily) a instância compartilhada do serviço."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Descarta a instância singleton — útil após atualizar o checkpoint."""
        cls._instance = None

    # ------------------------------------------------------------------
    # Embedding generation
    # ------------------------------------------------------------------

    def generate_embedding(self, image: Image.Image) -> list[float]:
        """
        Gera um embedding normalizado L2 a partir de uma imagem PIL.

        Steps:
            1. Converte para RGB
            2. Aplica o preprocess do CLIP (resize 224px + normalização)
            3. Forward pass pelo encoder visual (base ou fine-tunado)
            4. Normalização L2 — cosine similarity == produto interno
            5. Retorna como lista Python para pgvector

        Returns:
            Lista de 512 floats.
        """
        if image.mode != "RGB":
            image = image.convert("RGB")

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
        return "CLIP ViT-B/32 (fine-tuned)" if self._finetuned else "CLIP ViT-B/32"
