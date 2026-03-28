import axios from "axios"
import type { ApiError } from "@/types/api"

export const apiClient = axios.create({
  // Em produção/rede local usa o proxy reverso (/api) para evitar mixed content.
  // Em dev puro (sem proxy) pode apontar diretamente para o backend.
  baseURL: process.env.NEXT_PUBLIC_API_URL ?? "/api",
  timeout: 60_000, // 60s — embeddings podem demorar na primeira requisição
  headers: {
    "X-API-Key": process.env.NEXT_PUBLIC_API_KEY ?? "",
  },
})

// Normaliza erros da API FastAPI ({ detail: string }) para mensagens legíveis
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    const detail = (error.response?.data as ApiError)?.detail
    if (detail) error.message = typeof detail === "string" ? detail : JSON.stringify(detail)
    return Promise.reject(error)
  }
)
