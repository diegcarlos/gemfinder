"""
Faz upload do modelo fine-tunado para o Cloudflare R2.

Uso:
    python scripts/upload_model.py
    python scripts/upload_model.py --path models/clip_jewelry.pt
    python scripts/upload_model.py --key models/clip_jewelry_v2.pt
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload do modelo fine-tunado para o R2.")
    parser.add_argument("--path", default="models/clip_jewelry.pt", help="Caminho local do modelo.")
    parser.add_argument("--key",  default="models/clip_jewelry.pt", help="Chave de destino no R2.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    local_path = Path(args.path)

    if not local_path.exists():
        print(f"[ERROR] Arquivo não encontrado: {local_path}", file=sys.stderr)
        sys.exit(1)

    client = boto3.client(
        "s3",
        endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
        region_name="auto",
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        config=Config(signature_version="s3v4"),
    )

    size = local_path.stat().st_size

    class Progress:
        def __init__(self):
            self.uploaded = 0
        def __call__(self, chunk):
            self.uploaded += chunk
            pct = self.uploaded / size * 100
            print(f"\r  {pct:.1f}% ({self.uploaded/1e6:.1f} MB / {size/1e6:.1f} MB)", end="", flush=True)

    print(f"Enviando '{local_path}' → R2:{args.key} ...")
    try:
        client.upload_file(str(local_path), os.environ["R2_BUCKET_NAME"], args.key, Callback=Progress())
        print(f"\n✓ Upload concluído: {size/1e6:.1f} MB")
    except (BotoCoreError, ClientError) as exc:
        print(f"\n[ERROR] Falha no upload: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
