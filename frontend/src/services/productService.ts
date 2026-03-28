import { apiClient } from "@/http/apiClient"
import { ENDPOINTS } from "@/constants/endpoints"
import type { HealthResponse, ImageType, ProductImageSavedResponse } from "@/types/api"

/**
 * Faz upload de uma imagem para um produto específico.
 *
 * Se image_type for "clean", a API gera o embedding CLIP e cria/atualiza o produto.
 * Para outros tipos, apenas salva a imagem associada ao produto existente.
 */
export async function saveProductImage(
  file: File,
  productId: string,
  imageType: ImageType = "clean",
): Promise<ProductImageSavedResponse> {
  const form = new FormData()
  form.append("file", file)
  form.append("product_id", productId)
  form.append("image_type", imageType)

  const { data } = await apiClient.post<ProductImageSavedResponse>(
    ENDPOINTS.PRODUCTS_IMAGE,
    form,
  )
  return data
}

/** Verifica o status da API e o modelo carregado. */
export async function getHealth(): Promise<HealthResponse> {
  const { data } = await apiClient.get<HealthResponse>(ENDPOINTS.HEALTH)
  return data
}
