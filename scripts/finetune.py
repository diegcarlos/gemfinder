"""
finetune.py — Fine-tuning do CLIP ViT-B/32 para joias usando SimCLR.

Estratégia (auto-supervisionada — sem labels):
  • Pares positivos: duas vistas aumentadas da mesma imagem fundobranco
  • Loss: NT-Xent (InfoNCE) com temperatura 0.07
  • Camadas descongeladas: últimos 4 blocos do ViT + LayerNorm + projection
    (resto permanece congelado para preservar features gerais do CLIP)
  • Saída: 512-dim, mesmo formato do CLIP base — sem mudança de schema

Uso:
    python scripts/finetune.py catalago.zip
    python scripts/finetune.py ./imagens/ --epochs 30 --batch-size 64 --lr 5e-6
    python scripts/finetune.py catalago.zip --resume models/clip_jewelry.pt
"""

import argparse
import io
import math
import sys
import time
import zipfile
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import open_clip
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from app.services.image_service import is_supported_image, parse_filename

# ── Constantes ────────────────────────────────────────────────────────────────

# Normalização específica do CLIP (diferente do ImageNet)
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]

# Número de blocos transformer a descongelar (do final para o início)
UNFREEZE_BLOCKS = 4

OUTPUT_DIR = Path("models")


# ── Augmentation Pipeline ─────────────────────────────────────────────────────

def build_augmentation() -> Callable:
    """
    Pipeline de augmentação calibrado para imagens de joias com fundo branco.

    Escolhas:
      • RandomResizedCrop  : simula diferentes enquadramentos / zoom
      • RandomHorizontalFlip: joias são geralmente simétricas
      • ColorJitter (suave): pequenas variações de iluminação / câmera
      • RandomGrayscale 10%: robustez a imagens sem cor
      • Gaussian blur 20%  : robustez a foco ligeiramente diferente
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0), ratio=(0.85, 1.18)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15, hue=0.04),
        transforms.RandomGrayscale(p=0.10),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=9)], p=0.20),
        transforms.ToTensor(),
        transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])


# ── Dataset ───────────────────────────────────────────────────────────────────

class JewelryDataset(Dataset):
    """
    Carrega imagens fundobranco de um diretório ou arquivo ZIP.

    Retorna dois tensores aumentados da mesma imagem (vista1, vista2)
    para uso no loss NT-Xent.
    """

    def __init__(self, source: str, augment: Callable) -> None:
        self.augment = augment
        self.images: list[bytes] = []  # bytes brutos de cada imagem
        self._load(source)

    def _load(self, source: str) -> None:
        path = Path(source)

        if path.suffix.lower() == ".zip":
            self._load_zip(path)
        elif path.is_dir():
            self._load_dir(path)
        else:
            print(f"[ERROR] Fonte inválida: {source}", file=sys.stderr)
            sys.exit(1)

        if not self.images:
            print("[ERROR] Nenhuma imagem fundobranco encontrada.", file=sys.stderr)
            sys.exit(1)

        print(f"  → {len(self.images)} imagens fundobranco carregadas.")

    def _load_zip(self, path: Path) -> None:
        with zipfile.ZipFile(path, "r") as zf:
            for entry in zf.namelist():
                if entry.endswith("/"):
                    continue
                filename = Path(entry).name
                if not is_supported_image(filename):
                    continue
                parsed = parse_filename(filename)
                if parsed and parsed.image_type == "clean":
                    self.images.append(zf.read(entry))

    def _load_dir(self, path: Path) -> None:
        for file in sorted(path.rglob("*")):
            if not is_supported_image(file.name):
                continue
            parsed = parse_filename(file.name)
            if parsed and parsed.image_type == "clean":
                self.images.append(file.read_bytes())

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        raw = self.images[idx]
        image = Image.open(io.BytesIO(raw)).convert("RGB")
        # Duas vistas aumentadas independentes da mesma imagem = par positivo
        return self.augment(image), self.augment(image)


# ── Loss ──────────────────────────────────────────────────────────────────────

def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """
    NT-Xent (Normalized Temperature-scaled Cross Entropy).

    Trata (z1[i], z2[i]) como par positivo.
    Todas as outras combinações dentro do batch são negativas.

    Args:
        z1, z2: Embeddings das duas vistas — shape (N, D), já normalizados L2.
        temperature: Escala da distribuição de similaridade (padrão CLIP: 0.07).

    Returns:
        Scalar loss.
    """
    N = z1.size(0)
    # Concatena as 2N amostras
    z = torch.cat([z1, z2], dim=0)                    # (2N, D)
    sim = torch.mm(z, z.T) / temperature               # (2N, 2N)

    # Mascara a diagonal (auto-similaridade → -inf para excluir do softmax)
    mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(mask, float("-inf"))

    # Para cada amostra i em [0, N), o positivo é i+N; para i em [N, 2N), é i-N
    labels = torch.cat([
        torch.arange(N, 2 * N, device=z.device),
        torch.arange(0, N, device=z.device),
    ])

    return F.cross_entropy(sim, labels)


# ── Model setup ───────────────────────────────────────────────────────────────

def prepare_model(device: torch.device, resume: str | None):
    """
    Carrega CLIP ViT-B/32 e configura quais camadas serão treinadas.

    Camadas descongeladas:
      • Últimos UNFREEZE_BLOCKS blocos do ViT (resblocks[-4:])
      • LayerNorm final (ln_post)
      • Projeção de saída (proj)

    O restante permanece congelado — preserva os features gerais do CLIP
    e acelera o treino.

    Returns:
        (model, start_epoch, best_loss)
    """
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")

    # Congela tudo
    for param in model.parameters():
        param.requires_grad = False

    # Descongela os últimos N blocos do ViT
    for block in model.visual.transformer.resblocks[-UNFREEZE_BLOCKS:]:
        for param in block.parameters():
            param.requires_grad = True

    # Descongela LayerNorm final e projection do encoder visual
    for param in model.visual.ln_post.parameters():
        param.requires_grad = True
    if model.visual.proj is not None:
        model.visual.proj.requires_grad = True

    model = model.to(device)

    start_epoch = 0
    best_loss = float("inf")

    # Retoma checkpoint anterior se fornecido
    if resume:
        ckpt = torch.load(resume, map_location=device, weights_only=True)
        model.visual.load_state_dict(ckpt["visual_state_dict"])
        start_epoch = ckpt.get("epoch", 0)
        best_loss = ckpt.get("loss", float("inf"))
        print(f"  → Retomando do checkpoint (época {start_epoch}, loss {best_loss:.4f})")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  → Parâmetros treináveis: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    return model, start_epoch, best_loss


# ── Training ──────────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  Fine-tuning CLIP ViT-B/32 para joias")
    print(f"  Device   : {device}")
    print(f"  Source   : {args.source}")
    print(f"  Épocas   : {args.epochs}")
    print(f"  Batch    : {args.batch_size}")
    print(f"  LR       : {args.lr}")
    print(f"  Output   : {args.output}")
    print(f"{'='*60}\n")

    # ── Dataset e DataLoader ──────────────────────────────────────────
    print("Carregando imagens...")
    augment = build_augmentation()
    dataset = JewelryDataset(args.source, augment)

    if len(dataset) < args.batch_size:
        print(
            f"[WARN] Batch size ({args.batch_size}) > dataset ({len(dataset)}). "
            f"Reduzindo para {len(dataset)}."
        )
        args.batch_size = len(dataset)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,   # Windows: evita problemas com multiprocessing
        pin_memory=device.type == "cuda",
        drop_last=True,  # garante batches completos para o loss
    )

    # ── Modelo ────────────────────────────────────────────────────────
    print("Preparando modelo...")
    model, start_epoch, best_loss = prepare_model(device, args.resume)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=1e-4,
    )

    # Scheduler: warmup linear nos primeiros 10% dos steps + cosine decay
    total_steps   = args.epochs * len(loader)
    warmup_steps  = max(1, int(0.10 * total_steps))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.05, 0.5 * (1 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Loop de treino ────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(exist_ok=True)
    global_step = start_epoch * len(loader)

    for epoch in range(start_epoch, start_epoch + args.epochs):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        bar = tqdm(loader, desc=f"Época {epoch+1}/{start_epoch + args.epochs}", leave=False)

        for view1, view2 in bar:
            view1 = view1.to(device)
            view2 = view2.to(device)

            # Forward: extrai features das duas vistas e normaliza L2
            z1 = model.encode_image(view1)
            z1 = z1 / z1.norm(dim=-1, keepdim=True)

            z2 = model.encode_image(view2)
            z2 = z2 / z2.norm(dim=-1, keepdim=True)

            loss = nt_xent_loss(z1, z2, temperature=args.temperature)

            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping para estabilidade
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=1.0
            )

            optimizer.step()
            scheduler.step()
            global_step += 1

            epoch_loss += loss.item()
            bar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

        avg_loss = epoch_loss / len(loader)
        elapsed  = time.time() - t0
        print(
            f"  Época {epoch+1:3d} | loss {avg_loss:.4f} | "
            f"lr {scheduler.get_last_lr()[0]:.2e} | {elapsed:.0f}s"
        )

        # Salva o melhor checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            _save_checkpoint(model, epoch + 1, best_loss, args.output)
            print(f"  ✓ Checkpoint salvo → {args.output}  (melhor loss: {best_loss:.4f})")

    print(f"\nTreino concluído. Melhor loss: {best_loss:.4f}")
    print(f"Checkpoint: {args.output}")
    print("\nPróximo passo: rode o reindex para regenerar os embeddings no banco:")
    print("  python scripts/reindex.py")


def _save_checkpoint(model, epoch: int, loss: float, output: str) -> None:
    """Salva apenas o state_dict do encoder visual — arquivo compacto e direto ao ponto."""
    torch.save(
        {
            "visual_state_dict": model.visual.state_dict(),
            "epoch": epoch,
            "loss": loss,
            "config": {"model": "ViT-B-32", "pretrained": "openai", "unfreeze_blocks": UNFREEZE_BLOCKS},
        },
        output,
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tuning do CLIP ViT-B/32 para busca visual de joias (SimCLR).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("source", help="ZIP ou diretório com imagens fundobranco.")
    parser.add_argument("--epochs",      type=int,   default=20,                          help="Número de épocas.")
    parser.add_argument("--batch-size",  type=int,   default=32,                          help="Tamanho do batch.")
    parser.add_argument("--lr",          type=float, default=1e-5,                        help="Learning rate inicial.")
    parser.add_argument("--temperature", type=float, default=0.07,                        help="Temperatura do NT-Xent.")
    parser.add_argument("--output",      type=str,   default=str(OUTPUT_DIR / "clip_jewelry.pt"), help="Caminho do checkpoint.")
    parser.add_argument("--resume",      type=str,   default=None,                        help="Retomar de checkpoint existente.")
    return parser.parse_args()


if __name__ == "__main__":
    from tqdm import tqdm  # importado aqui para erro claro se não instalado
    train(parse_args())
