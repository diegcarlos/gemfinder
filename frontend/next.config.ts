import type { NextConfig } from "next"

// API_URL é lida em runtime pelo servidor Next.js (não baked no bundle do cliente).
// Em Docker: defina API_URL=http://backend:5000 no docker-compose.
// Em dev local: usa NEXT_PUBLIC_API_URL do .env.local como fallback.
const API_URL =
  process.env.API_URL ??
  process.env.NEXT_PUBLIC_API_URL ??
  "http://localhost:5000"

const config: NextConfig = {
  // Build standalone — necessário para o Dockerfile multi-stage (node server.js)
  output: "standalone",

  // Imagens R2 são presignadas — desabilita otimização para URLs externas com query params
  images: {
    unoptimized: true,
  },

  // Proxy reverso: requisições /api/* são repassadas ao backend FastAPI.
  // Evita mixed content (HTTPS → HTTP) e expõe apenas uma porta no cliente.
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: `${API_URL}/:path*`,
      },
    ]
  },
}

export default config
