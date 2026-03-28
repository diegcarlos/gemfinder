#!/bin/sh
# Entrypoint do container backend.
# 1. Baixa o modelo fine-tunado do R2 (se ainda não estiver em disco).
# 2. Sobe o servidor uvicorn.
set -e

echo "[entrypoint] Verificando modelo fine-tunado..."
python scripts/download_model.py

echo "[entrypoint] Iniciando uvicorn..."
exec uvicorn app.main:app --host 0.0.0.0 --port 5000 --workers 1
