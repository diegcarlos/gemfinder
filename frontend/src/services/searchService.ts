import { apiClient } from "@/http/apiClient"
import { ENDPOINTS } from "@/constants/endpoints"
import type { SearchResponse } from "@/types/api"

/** Envia uma imagem e retorna os produtos mais similares do catálogo. */
export async function searchSimilar(file: File): Promise<SearchResponse> {
  const form = new FormData()
  form.append("file", file)

  const { data } = await apiClient.post<SearchResponse>(ENDPOINTS.SEARCH, form)
  return data
}
