export interface ImagesByType {
  clean: string[]
  environment: string[]
  person: string[]
}

export interface SearchResult {
  product_id: string
  name: string
  similarity: number   // 0.0 (dissimilar) → 1.0 (idêntico)
  main_image: string   // URL presignada da imagem clean (principal)
  images: ImagesByType // Todas as URLs presignadas por tipo
}

export interface SearchResponse {
  results: SearchResult[]
  total: number
}

export interface ProductImageSavedResponse {
  id: string
  product_id: string
  type: string
  url: string  // URL presignada válida por 1 hora
}

export type ImageType = 'clean' | 'environment' | 'person'

export interface HealthResponse {
  status: string
  model: string
  embedding_dim: number
}

export interface ApiError {
  detail: string
}
