"""
import_zip.py — Pipeline de importação em lote para o GemFinder.

Uso (a partir da raiz do projeto):
    python scripts/import_zip.py path/to/images.zip

Convenção de nomes esperada:
    {product_id}_fundobranco.{ext}   → imagem clean  (gera embedding)
    {product_id}_ambiente.{ext}      → imagem de ambiente
    {product_id}_pessoa.{ext}        → imagem com pessoa

Comportamento:
    1. Lê e agrupa todas as imagens do ZIP por product_id
    2. Ignora produtos sem imagem de fundo branco (loga aviso)
    3. Gera embedding CLIP apenas da imagem fundobranco
    4. Faz upload de TODAS as variantes para o R2
    5. Salva o produto e todas as suas imagens no banco
    6. Idempotente: produtos já importados são pulados automaticamente
"""

import argparse
import asyncio
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import select

from app.database import AsyncSessionLocal, init_db
from app.models import ImageType, Product, ProductImage
from app.services.embedding_service import EmbeddingService
from app.services.image_service import (
    is_supported_image,
    load_image_from_bytes,
    normalise_image,
    parse_filename,
)
from app.services.s3_service import S3Service


class ImageEntry(NamedTuple):
    """Referência a uma imagem dentro do ZIP."""
    zip_entry: str   # caminho dentro do ZIP
    filename: str    # nome do arquivo (sem diretório)
    image_type: str  # "clean" | "environment" | "person"


async def import_zip(zip_path: str) -> None:
    """
    Processa o ZIP agrupando imagens por product_id e importando cada produto.

    Args:
        zip_path: Caminho absoluto ou relativo para o arquivo .zip.
    """
    zip_file = Path(zip_path)
    if not zip_file.exists():
        print(f"[ERROR] Arquivo não encontrado: {zip_path}")
        sys.exit(1)
    if zip_file.suffix.lower() != ".zip":
        print(f"[ERROR] Esperado um arquivo .zip, recebeu: {zip_file.suffix}")
        sys.exit(1)

    await init_db()

    embedding_svc = EmbeddingService.get_instance()
    s3_svc = S3Service()

    # ── 1. Ler e agrupar todas as entradas do ZIP por product_id ──────
    # { product_id: [ImageEntry, ...] }
    groups: dict[str, list[ImageEntry]] = defaultdict(list)
    skipped_unrecognized = 0

    with zipfile.ZipFile(zip_file, "r") as zf:
        for entry in zf.namelist():
            if entry.endswith("/"):
                continue  # ignora diretórios
            filename = Path(entry).name
            if not is_supported_image(filename):
                continue

            parsed = parse_filename(filename)
            if parsed is None:
                # Arquivo não segue a convenção de nomenclatura
                print(f"  [WARN] Ignorado (nome não reconhecido): {filename}")
                skipped_unrecognized += 1
                continue

            groups[parsed.product_id].append(
                ImageEntry(
                    zip_entry=entry,
                    filename=filename,
                    image_type=parsed.image_type,
                )
            )

    total_products = len(groups)
    print(f"Encontrados {total_products} produto(s) no arquivo {zip_file.name}")
    if skipped_unrecognized:
        print(f"  [{skipped_unrecognized} arquivo(s) ignorado(s) por nome não reconhecido]")

    ok = skipped = errors = 0

    async with AsyncSessionLocal() as session:
        with zipfile.ZipFile(zip_file, "r") as zf:
            for product_id, entries in groups.items():

                # ── 2. Verificar se já foi importado ──────────────────
                existing = await session.execute(
                    select(Product).where(Product.id == product_id).limit(1)
                )
                if existing.scalar_one_or_none() is not None:
                    print(f"  [SKIP] {product_id} — já importado")
                    skipped += 1
                    continue

                # ── 3. Garantir que existe imagem fundobranco ─────────
                clean_entries = [e for e in entries if e.image_type == "clean"]
                if not clean_entries:
                    print(
                        f"  [WARN] {product_id} — sem imagem fundobranco (clean), ignorado"
                    )
                    skipped += 1
                    continue

                # Usa a primeira imagem clean encontrada para o embedding
                clean_entry = clean_entries[0]

                try:
                    # ── 4. Processar imagem clean ─────────────────────
                    clean_bytes = zf.read(clean_entry.zip_entry)
                    clean_image = load_image_from_bytes(clean_bytes)
                    clean_image = normalise_image(clean_image)

                    # ── 5. Gerar embedding CLIP da imagem clean ───────
                    embedding = await embedding_svc.generate_embedding_async(clean_image)

                    # ── 6. Upload de TODAS as variantes para o R2 ─────
                    uploaded: list[tuple[str, str]] = []  # [(s3_key, image_type), ...]
                    main_image_url: str | None = None

                    for entry in entries:
                        raw = zf.read(entry.zip_entry)
                        s3_key = await s3_svc.upload_image_async(raw, entry.filename)
                        uploaded.append((s3_key, entry.image_type))
                        if entry.filename == clean_entry.filename:
                            main_image_url = s3_key

                    if main_image_url is None:
                        main_image_url = uploaded[0][0]

                    # ── 7. Persistir produto e imagens no banco ───────
                    product = Product(
                        id=product_id,
                        name=product_id,
                        main_image_url=main_image_url,
                        embedding=embedding,
                    )
                    session.add(product)
                    await session.flush()  # garante FK válida antes de inserir imagens

                    for s3_key, img_type in uploaded:
                        session.add(ProductImage(
                            product_id=product_id,
                            type=ImageType(img_type),
                            url=s3_key,
                        ))

                    await session.commit()

                    types_summary = ", ".join(sorted({e.image_type for e in entries}))
                    print(f"  [OK]   {product_id} — {len(entries)} imagem(ns): [{types_summary}]")
                    ok += 1

                except Exception as exc:
                    await session.rollback()
                    print(f"  [ERR]  {product_id}: {exc}")
                    errors += 1

    print(
        f"\nConcluído. "
        f"Importados: {ok} | Pulados: {skipped} | Erros: {errors} "
        f"| Total: {total_products}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Importação em lote de imagens de joalheria para o GemFinder."
    )
    parser.add_argument(
        "zip_path",
        help="Caminho para o arquivo .zip contendo as imagens dos produtos.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(import_zip(args.zip_path))
