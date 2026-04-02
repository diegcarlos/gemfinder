"""
finetune.py — Fine-tuning supervisionado do CLIP ViT-B/32 para joias.

Estratégia (metric learning supervisionado):
  • Dataset agrupado por product_id — cada produto é uma "classe"
  • Loss: Triplet Margin Loss com batch-hard mining
      - Anchor:   imagem do produto A
      - Positive: outra imagem do produto A (ângulo diferente) ou vista augmentada
      - Negative: imagem do produto mais similar (hard negative) entre os outros
  • Preprocessing: crop ao objeto + padding quadrado (equaliza domínio visual)
  • Augmentação calibrada para joias sobre fundo branco
  • Avaliação: Top-1 e Top-5 accuracy por época (hold-out 20% dos produtos)
  • Salva melhor checkpoint por validation accuracy (não por loss de treino)
  • Suporte a checkpoint com versionamento semântico

Uso:
    python scripts/finetune.py catalago.zip
    python scripts/finetune.py ./imagens/ --epochs 30 --batch-size 64 --lr 5e-6
    python scripts/finetune.py catalago.zip --resume models/clip_jewelry.pt
    python scripts/finetune.py catalago.zip --no-preprocess   # desativa crop/pad
"""

import argparse
import io
import math
import random
import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import open_clip
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms
from tqdm import tqdm

from app.services.image_service import is_supported_image, parse_filename

# ── Constantes ────────────────────────────────────────────────────────────────

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]

# Blocos do ViT a descongelar (do final para o início)
UNFREEZE_BLOCKS = 4

# Versão do modelo — incrementar ao mudar arquitetura ou estratégia de treino
MODEL_VERSION = "v2-triplet"

OUTPUT_DIR = Path("models")


# ── Augmentation ──────────────────────────────────────────────────────────────

def build_augmentation(strong: bool = True) -> Callable:
    """
    Pipeline de augmentação para imagens de joias.

    strong=True  → usado durante treino (mais variação)
    strong=False → usado durante avaliação (apenas normalização)
    """
    if not strong:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ])

    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.15, hue=0.03),
        transforms.RandomGrayscale(p=0.08),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=5)], p=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess_image_bytes(raw: bytes, apply_preprocessing: bool = True) -> Image.Image:
    """
    Decodifica bytes e aplica crop/pad para centralizar o objeto.

    A remoção de fundo NÃO é aplicada aqui (imagens fundobranco já têm fundo
    claro; o crop detecta pixels não-brancos e centraliza o objeto).
    """
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    if not apply_preprocessing:
        return image

    from app.services.image_preprocessing_service import crop_to_object, pad_to_square
    image = crop_to_object(image)
    image = pad_to_square(image)
    return image


# ── Dataset ───────────────────────────────────────────────────────────────────

class JewelryTripletDataset(Dataset):
    """
    Dataset agrupado por product_id para metric learning.

    Estrutura interna:
        product_ids: lista de IDs únicos
        images_by_product: dict product_id → lista de PIL Images pré-processadas

    Cada __getitem__ retorna (label_index, image_tensor).
    Os triplets são formados dinamicamente no loop de treino via batch-hard mining.
    """

    def __init__(
        self,
        source: str,
        augment: Callable,
        apply_preprocessing: bool = True,
        min_images_per_product: int = 1,
    ) -> None:
        self.augment = augment
        self.apply_preprocessing = apply_preprocessing

        # product_id → lista de PIL Images
        raw_groups: dict[str, list[Image.Image]] = defaultdict(list)
        self._load(source, raw_groups)

        # Filtra produtos com pelo menos min_images_per_product imagem
        self.product_ids = sorted(
            pid for pid, imgs in raw_groups.items()
            if len(imgs) >= min_images_per_product
        )

        if not self.product_ids:
            print("[ERROR] Nenhum produto com imagens suficientes encontrado.", file=sys.stderr)
            sys.exit(1)

        self.images_by_product = {pid: raw_groups[pid] for pid in self.product_ids}
        self.label_to_idx = {pid: i for i, pid in enumerate(self.product_ids)}

        # Índice flat: (product_id, image_index_within_product)
        self.samples: list[tuple[str, int]] = [
            (pid, i)
            for pid in self.product_ids
            for i in range(len(self.images_by_product[pid]))
        ]

        total_images = sum(len(v) for v in self.images_by_product.values())
        multi = sum(1 for v in self.images_by_product.values() if len(v) > 1)
        print(
            f"  → {len(self.product_ids)} produtos | {total_images} imagens "
            f"({multi} com múltiplas imagens)"
        )

    def _load(self, source: str, groups: dict[str, list[Image.Image]]) -> None:
        path = Path(source)
        if path.suffix.lower() == ".zip":
            self._load_zip(path, groups)
        elif path.is_dir():
            self._load_dir(path, groups)
        else:
            print(f"[ERROR] Fonte inválida: {source}", file=sys.stderr)
            sys.exit(1)

    def _load_zip(self, path: Path, groups: dict[str, list[Image.Image]]) -> None:
        with zipfile.ZipFile(path, "r") as zf:
            entries = [e for e in zf.namelist() if not e.endswith("/")]
            for entry in tqdm(entries, desc="Carregando ZIP", leave=False):
                filename = Path(entry).name
                if not is_supported_image(filename):
                    continue
                parsed = parse_filename(filename)
                # Usa apenas imagens fundobranco como âncora principal
                if parsed and parsed.image_type == "clean":
                    raw = zf.read(entry)
                    try:
                        img = preprocess_image_bytes(raw, self.apply_preprocessing)
                        groups[parsed.product_id].append(img)
                    except Exception:
                        pass

    def _load_dir(self, path: Path, groups: dict[str, list[Image.Image]]) -> None:
        files = [f for f in sorted(path.rglob("*")) if is_supported_image(f.name)]
        for file in tqdm(files, desc="Carregando diretório", leave=False):
            parsed = parse_filename(file.name)
            if parsed and parsed.image_type == "clean":
                try:
                    img = preprocess_image_bytes(file.read_bytes(), self.apply_preprocessing)
                    groups[parsed.product_id].append(img)
                except Exception:
                    pass

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[int, torch.Tensor]:
        pid, img_idx = self.samples[idx]
        image = self.images_by_product[pid][img_idx]
        label = self.label_to_idx[pid]
        return label, self.augment(image)

    def split_train_val(self, val_ratio: float = 0.2, seed: int = 42):
        """
        Divide o dataset em treino e validação por produto.

        Retorna dois datasets independentes que não compartilham produtos,
        garantindo que a avaliação teste generalização real.
        """
        rng = random.Random(seed)
        all_pids = self.product_ids.copy()
        rng.shuffle(all_pids)

        n_val = max(1, int(len(all_pids) * val_ratio))
        val_pids  = set(all_pids[:n_val])
        train_pids = set(all_pids[n_val:])

        return _SubsetDataset(self, train_pids), _SubsetDataset(self, val_pids)


class _SubsetDataset(Dataset):
    """Dataset derivado que inclui apenas um subset de produtos."""

    def __init__(self, parent: JewelryTripletDataset, product_ids: set[str]) -> None:
        self.augment = parent.augment
        self.product_ids = sorted(product_ids)
        self.images_by_product = {pid: parent.images_by_product[pid] for pid in self.product_ids}
        self.label_to_idx = {pid: i for i, pid in enumerate(self.product_ids)}
        self.samples: list[tuple[str, int]] = [
            (pid, i)
            for pid in self.product_ids
            for i in range(len(self.images_by_product[pid]))
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[int, torch.Tensor]:
        pid, img_idx = self.samples[idx]
        image = self.images_by_product[pid][img_idx]
        label = self.label_to_idx[pid]
        return label, self.augment(image)


# ── Loss: Triplet Margin com Batch-Hard Mining ────────────────────────────────

def batch_hard_triplet_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.3,
) -> tuple[torch.Tensor, int]:
    """
    Triplet Margin Loss com batch-hard mining.

    Para cada amostra como âncora:
      • Hardest positive: amostra do mesmo produto com MAIOR distância
      • Hardest negative: amostra de produto diferente com MENOR distância

    Esta estratégia força o modelo a aprender separações difíceis, acelerando
    a convergência comparada ao mining aleatório ou NT-Xent sem rótulos.

    Args:
        embeddings: Tensor (N, D) normalizado L2.
        labels:     Tensor (N,) com índice do produto para cada amostra.
        margin:     Margem mínima entre d(a,p) e d(a,n). Padrão: 0.3.

    Returns:
        (loss scalar, n_valid_triplets)
    """
    # Matriz de distâncias euclidianas ao quadrado: (N, N)
    # Com embeddings L2-normalizados: ||a-b||² = 2 - 2*cos(a,b)
    dot = torch.mm(embeddings, embeddings.T)
    sq_norm = dot.diagonal()
    dist_sq = sq_norm.unsqueeze(1) - 2 * dot + sq_norm.unsqueeze(0)
    dist_sq = dist_sq.clamp(min=0.0)
    dist = torch.sqrt(dist_sq + 1e-8)

    # Máscaras de pares válidos
    labels_eq  = labels.unsqueeze(0) == labels.unsqueeze(1)   # mesmo produto
    labels_neq = ~labels_eq

    # Remove a diagonal (auto-comparação)
    eye = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    valid_pos_mask = labels_eq  & ~eye
    valid_neg_mask = labels_neq

    # Hardest positive: maior distância entre amostras do mesmo produto
    # Máscaramos os pares inválidos com -inf antes do max
    hardest_pos = (dist * valid_pos_mask.float() + (~valid_pos_mask).float() * -1e9).max(dim=1).values

    # Hardest negative: menor distância entre amostras de produtos diferentes
    # Máscaramos pares inválidos com +inf antes do min
    hardest_neg = (dist * valid_neg_mask.float() + (~valid_neg_mask).float() * 1e9).min(dim=1).values

    # Só calcula loss para âncoras que têm pelo menos 1 positivo e 1 negativo válidos
    has_pos = valid_pos_mask.any(dim=1)
    has_neg = valid_neg_mask.any(dim=1)
    valid   = has_pos & has_neg

    if not valid.any():
        return torch.tensor(0.0, device=embeddings.device, requires_grad=True), 0

    loss = F.relu(hardest_pos[valid] - hardest_neg[valid] + margin).mean()
    return loss, int(valid.sum().item())


# ── Avaliação: Top-K Accuracy ─────────────────────────────────────────────────

@torch.no_grad()
def evaluate_top_k(
    model: torch.nn.Module,
    dataset: _SubsetDataset,
    device: torch.device,
    k_values: tuple[int, ...] = (1, 5),
    batch_size: int = 64,
) -> dict[str, float]:
    """
    Avalia Top-K accuracy no dataset de validação.

    Para cada produto, usa a primeira imagem como query e verifica se alguma
    das K mais similares pertence ao mesmo produto.

    Returns:
        Dict com "top1", "top5" (etc.) como chaves e accuracy [0,1] como valor.
    """
    model.eval()
    val_aug = build_augmentation(strong=False)

    # Gera embeddings para todas as imagens de validação (sem augmentação)
    all_embeddings: list[torch.Tensor] = []
    all_labels:     list[int] = []

    loader = DataLoader(
        _AugOverrideDataset(dataset, val_aug),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    for labels, images in loader:
        images = images.to(device)
        emb = model.encode_image(images)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        all_embeddings.append(emb.cpu())
        all_labels.extend(labels.tolist())

    embeddings = torch.cat(all_embeddings, dim=0)  # (N, D)
    labels_t   = torch.tensor(all_labels)

    # Similaridade coseno entre todos os pares
    sim_matrix = torch.mm(embeddings, embeddings.T)  # (N, N)

    # Mascara a diagonal (não conta auto-similaridade)
    eye = torch.eye(len(labels_t), dtype=torch.bool)
    sim_matrix[eye] = -2.0

    results: dict[str, float] = {}
    for k in k_values:
        if k >= len(labels_t):
            continue
        top_k_indices = sim_matrix.topk(k, dim=1).indices  # (N, K)
        top_k_labels  = labels_t[top_k_indices]             # (N, K)
        correct = (top_k_labels == labels_t.unsqueeze(1)).any(dim=1).float()
        results[f"top{k}"] = float(correct.mean().item())

    return results


class _AugOverrideDataset(Dataset):
    """Wrapper que substitui a augmentação de um dataset existente."""
    def __init__(self, dataset: _SubsetDataset, new_aug: Callable) -> None:
        self.dataset = dataset
        self.new_aug = new_aug

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[int, torch.Tensor]:
        pid, img_idx = self.dataset.samples[idx]
        image = self.dataset.images_by_product[pid][img_idx]
        label = self.dataset.label_to_idx[pid]
        return label, self.new_aug(image)


# ── Model setup ───────────────────────────────────────────────────────────────

def prepare_model(device: torch.device, resume: str | None):
    """
    Carrega CLIP ViT-B/32 e descongela os últimos UNFREEZE_BLOCKS blocos.

    Returns:
        (model, start_epoch, best_metric, version)
    """
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")

    # Congela tudo
    for param in model.parameters():
        param.requires_grad = False

    # Descongela os últimos N blocos do ViT
    for block in model.visual.transformer.resblocks[-UNFREEZE_BLOCKS:]:
        for param in block.parameters():
            param.requires_grad = True

    # Descongela LayerNorm final e projection
    for param in model.visual.ln_post.parameters():
        param.requires_grad = True
    if model.visual.proj is not None:
        model.visual.proj.requires_grad = True

    model = model.to(device)

    start_epoch  = 0
    best_metric  = 0.0  # Top-1 accuracy (quanto maior, melhor)

    if resume:
        ckpt = torch.load(resume, map_location=device, weights_only=True)
        model.visual.load_state_dict(ckpt["visual_state_dict"])
        start_epoch = ckpt.get("epoch", 0)
        best_metric = ckpt.get("best_top1", ckpt.get("loss", 0.0))
        if isinstance(best_metric, float) and best_metric > 1.0:
            # Era um loss (maior = pior) — reset para 0
            best_metric = 0.0
        print(f"  → Retomando checkpoint: época {start_epoch}, best_top1 {best_metric:.4f}")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  → Treináveis: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    return model, start_epoch, best_metric


# ── Checkpoint ────────────────────────────────────────────────────────────────

def save_checkpoint(
    model,
    epoch: int,
    train_loss: float,
    val_metrics: dict[str, float],
    output: str,
) -> None:
    """
    Salva checkpoint com metadados completos para rastreabilidade.

    Campos salvos:
        visual_state_dict  — pesos do encoder visual fine-tunado
        epoch              — época de treino
        loss               — loss médio de treino (compatibilidade com v1)
        best_top1          — Top-1 accuracy na validação
        val_metrics        — dict completo com top1, top5, etc.
        version            — versão semântica do modelo
        config             — hiperparâmetros de treino
    """
    torch.save(
        {
            "visual_state_dict": model.visual.state_dict(),
            "epoch":       epoch,
            "loss":        train_loss,
            "best_top1":   val_metrics.get("top1", 0.0),
            "val_metrics": val_metrics,
            "version":     MODEL_VERSION,
            "config": {
                "model":           "ViT-B-32",
                "pretrained":      "openai",
                "unfreeze_blocks": UNFREEZE_BLOCKS,
                "loss":            "triplet_batch_hard",
            },
        },
        output,
    )


# ── Training loop ─────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"  Fine-tuning CLIP ViT-B/32 — Triplet Batch-Hard")
    print(f"  Device      : {device}")
    print(f"  Source      : {args.source}")
    print(f"  Épocas      : {args.epochs}")
    print(f"  Batch       : {args.batch_size}")
    print(f"  LR          : {args.lr}")
    print(f"  Margem      : {args.margin}")
    print(f"  Preprocess  : {'sim' if args.preprocess else 'não'}")
    print(f"  Output      : {args.output}")
    print(f"  Versão      : {MODEL_VERSION}")
    print(f"{'='*60}\n")

    # ── Dataset ───────────────────────────────────────────────────────
    print("Carregando imagens...")
    augment = build_augmentation(strong=True)
    full_dataset = JewelryTripletDataset(
        args.source,
        augment,
        apply_preprocessing=args.preprocess,
        min_images_per_product=1,
    )

    train_ds, val_ds = full_dataset.split_train_val(val_ratio=0.2)
    print(f"  → Treino: {len(train_ds.product_ids)} produtos | Val: {len(val_ds.product_ids)} produtos")

    if len(train_ds) < args.batch_size:
        args.batch_size = max(2, len(train_ds))
        print(f"  [WARN] Batch reduzido para {args.batch_size}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
        drop_last=len(train_ds) > args.batch_size,
    )

    # ── Modelo ────────────────────────────────────────────────────────
    print("Preparando modelo...")
    model, start_epoch, best_top1 = prepare_model(device, args.resume)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=1e-4,
    )

    total_steps  = args.epochs * len(train_loader)
    warmup_steps = max(1, int(0.10 * total_steps))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.05, 0.5 * (1 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Loop de treino ─────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(exist_ok=True)
    global_step = start_epoch * len(train_loader)

    for epoch in range(start_epoch, start_epoch + args.epochs):
        model.train()
        epoch_loss    = 0.0
        total_triplets = 0
        t0 = time.time()

        bar = tqdm(
            train_loader,
            desc=f"Época {epoch + 1}/{start_epoch + args.epochs}",
            leave=False,
        )

        for labels, images in bar:
            labels = labels.to(device)
            images = images.to(device)

            # Extrai embeddings e normaliza L2
            emb = model.encode_image(images)
            emb = emb / emb.norm(dim=-1, keepdim=True)

            loss, n_valid = batch_hard_triplet_loss(emb, labels, margin=args.margin)

            if n_valid == 0:
                # Batch sem triplets válidos (todos os produtos são únicos no batch)
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=1.0
            )
            optimizer.step()
            scheduler.step()
            global_step += 1

            epoch_loss     += loss.item()
            total_triplets += n_valid
            bar.set_postfix(
                loss=f"{loss.item():.4f}",
                triplets=n_valid,
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
            )

        avg_loss = epoch_loss / max(1, len(train_loader))
        elapsed  = time.time() - t0

        # ── Avaliação ─────────────────────────────────────────────────
        val_metrics: dict[str, float] = {}
        if len(val_ds) >= 2:
            val_metrics = evaluate_top_k(model, val_ds, device, k_values=(1, 5))
            model.train()  # volta para modo treino

        top1 = val_metrics.get("top1", 0.0)
        top5 = val_metrics.get("top5", 0.0)

        print(
            f"  Época {epoch + 1:3d} | "
            f"loss {avg_loss:.4f} | "
            f"top1 {top1:.3f} | "
            f"top5 {top5:.3f} | "
            f"triplets {total_triplets} | "
            f"lr {scheduler.get_last_lr()[0]:.2e} | "
            f"{elapsed:.0f}s"
        )

        # Salva melhor modelo por Top-1 accuracy
        if top1 > best_top1 or (top1 == 0.0 and avg_loss < 1e9):
            if top1 > best_top1:
                best_top1 = top1
            save_checkpoint(model, epoch + 1, avg_loss, val_metrics, args.output)
            print(f"  ✓ Checkpoint salvo → {args.output}  (top1: {best_top1:.3f})")

    print(f"\nTreino concluído.")
    print(f"  Melhor Top-1: {best_top1:.3f}")
    print(f"  Checkpoint  : {args.output}")
    print("\nPróximo passo: rode o reindex para regenerar embeddings no banco:")
    print("  python scripts/reindex.py")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tuning CLIP ViT-B/32 com Triplet Loss para busca de joias.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("source",          help="ZIP ou diretório com imagens fundobranco.")
    parser.add_argument("--epochs",        type=int,   default=30,
                        help="Número de épocas.")
    parser.add_argument("--batch-size",    type=int,   default=64,
                        help="Tamanho do batch (recomendado ≥ 32 para batch-hard mining).")
    parser.add_argument("--lr",            type=float, default=5e-6,
                        help="Learning rate inicial.")
    parser.add_argument("--margin",        type=float, default=0.3,
                        help="Margem do Triplet Loss.")
    parser.add_argument("--output",        type=str,   default=str(OUTPUT_DIR / "clip_jewelry.pt"),
                        help="Caminho do checkpoint de saída.")
    parser.add_argument("--resume",        type=str,   default=None,
                        help="Retomar de checkpoint existente.")
    parser.add_argument("--no-preprocess", dest="preprocess", action="store_false",
                        help="Desativa crop/pad do objeto (mais rápido, mas menos preciso).")
    parser.set_defaults(preprocess=True)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
