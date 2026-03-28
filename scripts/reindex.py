"""
reindex.py — Regenera os embeddings de todos os produtos no banco de dados.

Execute após o fine-tuning para que os vetores armazenados reflitam o modelo
ajustado. O script baixa cada imagem principal (fundobranco) diretamente do R2,
re-gera o embedding com o modelo atual (fine-tunado ou base) e atualiza o banco.

Uso:
    python scripts/reindex.py
    python scripts/reindex.py --batch-size 32 --dry-run
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import select, update
from tqdm import tqdm

from app.database import AsyncSessionLocal, init_db
from app.models import Product
from app.services.embedding_service import EmbeddingService
from app.services.image_service import load_image_from_bytes, normalise_image
from app.services.s3_service import S3Service


async def reindex(batch_size: int, dry_run: bool) -> None:
    """
    Itera sobre todos os produtos e atualiza seus embeddings.

    Fluxo por produto:
        1. Baixa main_image_url do R2 via get_object (sem expiração)
        2. Decodifica e normaliza a imagem
        3. Gera embedding com o modelo carregado (fine-tunado se disponível)
        4. Atualiza a coluna embedding no banco

    Args:
        batch_size: Quantos produtos processar por commit.
        dry_run: Se True, apenas simula sem gravar no banco.
    """
    await init_db()

    svc = EmbeddingService.get_instance()
    s3  = S3Service()

    print(f"\n{'='*60}")
    print(f"  Reindexação de embeddings")
    print(f"  Modelo  : {svc.model_name}")
    print(f"  Dry-run : {dry_run}")
    print(f"  Batch   : {batch_size}")
    print(f"{'='*60}\n")

    async with AsyncSessionLocal() as session:
        # Conta o total para a barra de progresso
        result = await session.execute(select(Product))
        products = result.scalars().all()
        total = len(products)

        if total == 0:
            print("Nenhum produto encontrado no banco.")
            return

        print(f"Processando {total} produto(s)...\n")

        ok = errors = 0
        pending_updates: list[tuple[str, list[float]]] = []

        bar = tqdm(products, unit="produto")

        for product in bar:
            bar.set_description(f"{product.id}")
            try:
                # ── 1. Baixa imagem do R2 ─────────────────────────────
                raw_bytes = await s3.download_image_async(product.main_image_url)

                # ── 2. Decodifica e normaliza ─────────────────────────
                image = load_image_from_bytes(raw_bytes)
                image = normalise_image(image)

                # ── 3. Gera embedding ─────────────────────────────────
                embedding = await svc.generate_embedding_async(image)

                pending_updates.append((product.id, embedding))
                ok += 1

            except Exception as exc:
                tqdm.write(f"  [ERR] {product.id}: {exc}")
                errors += 1
                continue

            # ── 4. Commit em batches ──────────────────────────────────
            if len(pending_updates) >= batch_size:
                await _flush(session, pending_updates, dry_run)
                pending_updates.clear()

        # Flush final (resto que não completou um batch)
        if pending_updates:
            await _flush(session, pending_updates, dry_run)

    tag = "[DRY-RUN] " if dry_run else ""
    print(f"\n{tag}Concluído. Atualizados: {ok} | Erros: {errors} | Total: {total}")

    if dry_run:
        print("Nenhuma alteração gravada no banco (--dry-run).")


async def _flush(session, updates: list[tuple[str, list[float]]], dry_run: bool) -> None:
    """Aplica as atualizações de embedding em lote."""
    if dry_run:
        return

    for product_id, embedding in updates:
        await session.execute(
            update(Product)
            .where(Product.id == product_id)
            .values(embedding=embedding)
        )
    await session.commit()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenera embeddings de todos os produtos após fine-tuning.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="Número de produtos por commit no banco.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Simula o processo sem gravar no banco.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(reindex(args.batch_size, args.dry_run))
