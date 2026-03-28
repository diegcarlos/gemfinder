import axios from "axios"
import type { ApiError } from "@/types/api"

export const apiClient = axios.create({
  // Sempre usa o proxy reverso (/api) — o Route Handler server-side injeta o X-API-Key.
  baseURL: "/api",
  timeout: 60_000, // 60s — embeddings podem demorar na primeira requisição
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
