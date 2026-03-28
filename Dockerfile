FROM python:3.12-slim

WORKDIR /app

# Dependências de sistema:
#   libpq-dev + gcc → asyncpg (driver PostgreSQL)
#   libgl1 + libglib2.0 → Pillow com suporte a imagens variadas
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    gcc \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Instala dependências Python primeiro (camada cacheável)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pré-baixa os pesos do CLIP ViT-B/32 (~350 MB)
RUN python -c "import open_clip; open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai'); print('CLIP weights cached.')"

# Pré-baixa o modelo U2Net do rembg (~170 MB) para remoção de fundo
# Evita delay de ~30s na primeira requisição de busca em produção
RUN python -c "from rembg import new_session; new_session('u2net'); print('rembg U2Net cached.')"

# Copia o código da aplicação e scripts
COPY app/ ./app/
COPY scripts/ ./scripts/
COPY entrypoint.sh ./entrypoint.sh
RUN chmod +x entrypoint.sh

# Diretório para o modelo fine-tunado (persistido via volume no compose)
RUN mkdir -p models

EXPOSE 5000

# O entrypoint baixa o modelo do R2 na primeira inicialização, depois sobe o uvicorn
ENTRYPOINT ["./entrypoint.sh"]
