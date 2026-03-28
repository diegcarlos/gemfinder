"""
Baixa o modelo fine-tunado do Cloudflare R2 para o diretório local.

Uso:
    python scripts/download_model.py

Variáveis de ambiente necessárias (mesmas do .env):
    R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_ACCOUNT_ID, R2_BUCKET_NAME
    MODEL_R2_KEY  — chave do objeto no R2 (padrão: models/clip_jewelry.pt)
    FINETUNED_MODEL_PATH — destino local (padrão: models/clip_jewelry.pt)
"""

import os
import sys
from pathlib import Path

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

R2_ACCESS_KEY_ID     = os.environ["R2_ACCESS_KEY_ID"]
R2_SECRET_ACCESS_KEY = os.environ["R2_SECRET_ACCESS_KEY"]
R2_ACCOUNT_ID        = os.environ["R2_ACCOUNT_ID"]
R2_BUCKET_NAME       = os.environ["R2_BUCKET_NAME"]

MODEL_R2_KEY    = os.environ.get("MODEL_R2_KEY", "models/clip_jewelry.pt")
LOCAL_MODEL_PATH = Path(os.environ.get("FINETUNED_MODEL_PATH", "models/clip_jewelry.pt"))


def main() -> None:
    if LOCAL_MODEL_PATH.exists():
        print(f"[download_model] Modelo já presente em '{LOCAL_MODEL_PATH}', nada a fazer.")
        return

    LOCAL_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    endpoint_url = f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
    client = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        region_name="auto",
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        config=Config(signature_version="s3v4"),
    )

    print(f"[download_model] Baixando '{MODEL_R2_KEY}' do R2 → '{LOCAL_MODEL_PATH}' ...")
    try:
        client.download_file(R2_BUCKET_NAME, MODEL_R2_KEY, str(LOCAL_MODEL_PATH))
    except ClientError as exc:
        code = exc.response["Error"]["Code"]
        if code in ("NoSuchKey", "404"):
            print(
                f"[download_model] AVISO: chave '{MODEL_R2_KEY}' não encontrada no R2. "
                "O modelo fine-tunado não será carregado — usando CLIP base."
            )
            return
        print(f"[download_model] ERRO ao baixar modelo: {exc}", file=sys.stderr)
        sys.exit(1)
    except BotoCoreError as exc:
        print(f"[download_model] ERRO de conexão com R2: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"[download_model] Download concluído: {LOCAL_MODEL_PATH.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
