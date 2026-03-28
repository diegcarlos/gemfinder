import asyncio
import uuid
from pathlib import Path

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

from app.config import settings

# Tempo de validade padrão das URLs presignadas (1 hora)
PRESIGN_EXPIRES_IN: int = 3600


class S3Service:
    """
    Service para upload de imagens no Cloudflare R2 (S3-compatível).

    Como o bucket é privado, o acesso às imagens é feito via URLs presignadas
    geradas sob demanda com tempo de expiração configurável.

    O banco de dados armazena apenas a *chave* do objeto (ex: "products/uuid-ring.jpg"),
    não a URL completa. A URL assinada é gerada na hora de retornar ao cliente.
    """

    def __init__(self) -> None:
        endpoint_url = f"https://{settings.R2_ACCOUNT_ID}.r2.cloudflarestorage.com"

        self._client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            region_name="auto",
            aws_access_key_id=settings.R2_ACCESS_KEY_ID,
            aws_secret_access_key=settings.R2_SECRET_ACCESS_KEY,
            config=Config(signature_version="s3v4"),  # R2 exige SigV4
        )
        self.bucket = settings.R2_BUCKET_NAME

    # ------------------------------------------------------------------
    # Upload
    # ------------------------------------------------------------------

    def upload_image(self, image_bytes: bytes, filename: str) -> str:
        """
        Faz upload dos bytes da imagem para o R2 e retorna a *chave* do objeto.

        A chave é o caminho dentro do bucket (ex: "products/uuid-ring.jpg").
        Armazene a chave no banco — gere a URL assinada apenas ao servir ao cliente.

        Args:
            image_bytes: Conteúdo bruto da imagem (JPEG, PNG, etc.)
            filename:    Nome original do arquivo (usado para extensão e sufixo da chave).

        Returns:
            Chave S3 do objeto criado (ex: "products/abc123-ring.jpg").

        Raises:
            RuntimeError: Em caso de falha na API do R2.
        """
        ext = Path(filename).suffix.lower()
        content_type = _ext_to_content_type(ext)

        # Chave única dentro do bucket: "products/<uuid>-<filename>"
        key = f"products/{uuid.uuid4().hex}-{Path(filename).name}"

        try:
            self._client.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=image_bytes,
                ContentType=content_type,
            )
        except (BotoCoreError, ClientError) as exc:
            raise RuntimeError(f"Falha ao fazer upload de '{filename}' para o R2: {exc}") from exc

        return key  # retorna a chave, não a URL

    async def upload_image_async(self, image_bytes: bytes, filename: str) -> str:
        """Wrapper assíncrono para upload_image (executa em thread pool)."""
        return await asyncio.to_thread(self.upload_image, image_bytes, filename)

    # ------------------------------------------------------------------
    # Presigned URL
    # ------------------------------------------------------------------

    def presign(self, key_or_url: str, expires_in: int = PRESIGN_EXPIRES_IN) -> str:
        """
        Gera uma URL presignada (autenticada) para acessar um objeto privado no R2.

        Aceita tanto a chave S3 pura ("products/uuid-ring.jpg") quanto uma URL
        completa — útil para registros antigos salvos antes da migração para chaves.

        A URL expira após `expires_in` segundos (padrão: 1 hora).

        Args:
            key_or_url: Chave S3 ou URL completa do objeto.
            expires_in: Validade da URL em segundos.

        Returns:
            URL HTTPS temporária com assinatura SigV4.
        """
        key = _extract_key(key_or_url)
        return self._client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": key},
            ExpiresIn=expires_in,
        )

    async def presign_async(self, key: str, expires_in: int = PRESIGN_EXPIRES_IN) -> str:
        """Wrapper assíncrono para presign (executa em thread pool)."""
        return await asyncio.to_thread(self.presign, key, expires_in)

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    def download_image(self, key_or_url: str) -> bytes:
        """
        Baixa os bytes brutos de um objeto do R2 sem URL presignada.

        Uso interno (scripts de reindexação) — sem expiração de URL.

        Raises:
            RuntimeError: Em caso de falha na API do R2.
        """
        key = _extract_key(key_or_url)
        try:
            response = self._client.get_object(Bucket=self.bucket, Key=key)
            return response["Body"].read()
        except (BotoCoreError, ClientError) as exc:
            raise RuntimeError(f"Falha ao baixar '{key}' do R2: {exc}") from exc

    async def download_image_async(self, key_or_url: str) -> bytes:
        """Wrapper assíncrono para download_image (executa em thread pool)."""
        return await asyncio.to_thread(self.download_image, key_or_url)


def _extract_key(key_or_url: str) -> str:
    """
    Normaliza o valor armazenado em image_url para uma chave S3 pura.

    Registros antigos podem conter a URL completa do R2; registros novos
    já contêm apenas a chave. Ambos os formatos são tratados aqui.

    Exemplos:
        "products/abc-ring.jpg"
            → "products/abc-ring.jpg"
        "https://account.r2.cloudflarestorage.com/products/abc-ring.jpg"
            → "products/abc-ring.jpg"
    """
    if key_or_url.startswith("http://") or key_or_url.startswith("https://"):
        from urllib.parse import urlparse
        path = urlparse(key_or_url).path  # "/products/abc-ring.jpg"
        return path.lstrip("/")
    return key_or_url


def _ext_to_content_type(ext: str) -> str:
    """Mapeia extensão de arquivo para o MIME type correto."""
    mapping = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".gif": "image/gif",
        ".bmp": "image/bmp",
        ".tiff": "image/tiff",
    }
    return mapping.get(ext, "application/octet-stream")
