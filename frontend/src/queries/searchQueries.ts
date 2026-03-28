import { useMutation } from "@tanstack/react-query"
import { searchSimilar } from "@/services/searchService"

export function useSearchSimilar() {
  return useMutation({
    mutationFn: searchSimilar,
  })
}
