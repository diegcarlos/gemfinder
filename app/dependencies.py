from fastapi import Header, HTTPException, status

from app.config import settings


async def verify_api_key(x_api_key: str = Header(..., alias="X-API-Key")) -> None:
    """
    Dependência global de autenticação por API Key.

    Todas as rotas que incluírem esta dependência exigem o header:
        X-API-Key: <valor configurado em API_KEY no .env>

    Retorna 401 se o header estiver ausente ou inválido.
    Usar comparação de tempo constante (secrets.compare_digest) evita
    ataques de timing que tentam descobrir a chave por tempo de resposta.
    """
    import secrets

    if not secrets.compare_digest(x_api_key, settings.API_KEY):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key inválida ou ausente.",
            headers={"WWW-Authenticate": "ApiKey"},
        )
