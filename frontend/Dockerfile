# ── Stage 1: Dependências ────────────────────────────────────────────────────
FROM node:20-alpine AS deps
WORKDIR /app

COPY package.json package-lock.json* ./
RUN npm ci --frozen-lockfile

# ── Stage 2: Build ────────────────────────────────────────────────────────────
FROM node:20-alpine AS builder
WORKDIR /app

COPY --from=deps /app/node_modules ./node_modules
COPY . .

# NEXT_PUBLIC_API_KEY é baked no bundle do cliente — passa como build arg
ARG NEXT_PUBLIC_API_KEY
ENV NEXT_PUBLIC_API_KEY=$NEXT_PUBLIC_API_KEY

# NEXT_PUBLIC_API_URL não definido → cliente usa "/api" (proxy reverso do Next.js)
RUN npm run build

# ── Stage 3: Runner (imagem mínima de produção) ───────────────────────────────
FROM node:20-alpine AS runner
WORKDIR /app

ENV NODE_ENV=production

# Copia apenas os artefatos do build standalone
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static
COPY --from=builder /app/public ./public

EXPOSE 3000

# API_URL é lida em runtime pelo servidor Next.js para o proxy reverso
# Definida no docker-compose: API_URL=http://backend:5000
CMD ["node", "server.js"]
