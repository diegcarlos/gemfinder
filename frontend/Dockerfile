# ── Stage 1: Dependências ────────────────────────────────────────────────────
FROM oven/bun:1 AS deps
WORKDIR /app

COPY package.json bun.lock ./
RUN bun install --frozen-lockfile

# ── Stage 2: Build ────────────────────────────────────────────────────────────
FROM oven/bun:1 AS builder
WORKDIR /app

COPY --from=deps /app/node_modules ./node_modules
COPY . .

RUN bun run build

# ── Stage 3: Runner (imagem mínima de produção) ───────────────────────────────
FROM oven/bun:1 AS runner
WORKDIR /app

ENV NODE_ENV=production

# Copia apenas os artefatos do build standalone
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static
# public/ é opcional — só copia se existir
COPY --from=builder /app/public* ./public/

EXPOSE 3000

# API_URL é lida em runtime pelo servidor Next.js para o proxy reverso
# Definida no docker-compose: API_URL=http://backend:5000
CMD ["bun", "server.js"]
