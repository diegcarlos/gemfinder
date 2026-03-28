import { useMutation, useQuery } from "@tanstack/react-query"
import { saveProductImage, getHealth } from "@/services/productService"
import type { ImageType } from "@/types/api"

export function useSaveProductImage() {
  return useMutation({
    mutationFn: ({
      file,
      productId,
      imageType,
    }: {
      file: File
      productId: string
      imageType?: ImageType
    }) => saveProductImage(file, productId, imageType),
  })
}

export function useHealth() {
  return useQuery({
    queryKey: ["health"],
    queryFn: getHealth,
    refetchInterval: 30_000,
    retry: false,
  })
}
