import type { NextConfig } from "next"

const config: NextConfig = {
  // Build standalone — necessário para o Dockerfile multi-stage (node server.js)
  output: "standalone",

  // Imagens R2 são presignadas — desabilita otimização para URLs externas com query params
  images: {
    unoptimized: true,
  },
}

export default config
