"use client"

import { useState } from "react"
import { Search } from "lucide-react"
import { toast } from "sonner"
import { Button } from "@/components/ui/button"
import { ImageDropzone } from "@/components/ImageDropzone"
import { ProductCard, ProductCardSkeleton } from "@/components/ProductCard"
import { useSearchSimilar } from "@/queries/searchQueries"
import { useHealth } from "@/queries/productQueries"

export default function SearchPage() {
  const [image, setImage] = useState<File | null>(null)

  const { mutate, data, isPending, reset } = useSearchSimilar()
  const { data: health } = useHealth()

  function handleImageChange(file: File | null) {
    setImage(file)
    if (!file) reset()
  }

  function handleSearch() {
    if (!image) return
    mutate(image, {
      onError: (err) => toast.error(err.message || "Erro ao buscar produtos similares."),
    })
  }

  return (
    <main className="container max-w-5xl mx-auto py-12 px-4 space-y-10">
      {/* Hero */}
      <div className="text-center space-y-2">
        <h1 className="text-4xl font-bold tracking-tight">Busca por Similaridade</h1>
        <p className="text-muted-foreground">
          Envie a foto de uma joia e encontre as peças mais parecidas no catálogo
        </p>
        {health && (
          <p className="text-xs text-muted-foreground">
            Modelo: <span className="font-mono">{health.model}</span> · {health.embedding_dim} dims
          </p>
        )}
      </div>

      {/* Upload + Search */}
      <div className="max-w-sm mx-auto space-y-4">
        <ImageDropzone value={image} onChange={handleImageChange} className="h-64" />
        <Button
          className="w-full"
          size="lg"
          disabled={!image || isPending}
          onClick={handleSearch}
        >
          <Search className="mr-2 h-4 w-4" />
          {isPending ? "Buscando..." : "Buscar Similares"}
        </Button>
      </div>

      {/* Loading skeletons */}
      {isPending && (
        <section className="space-y-4">
          <div className="h-6 w-40 bg-muted animate-pulse rounded" />
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-4">
            {Array.from({ length: 5 }).map((_, i) => (
              <ProductCardSkeleton key={i} />
            ))}
          </div>
        </section>
      )}

      {/* Results */}
      {data && !isPending && (
        <section className="space-y-4">
          <h2 className="text-lg font-semibold">
            {data.results.length > 0
              ? `${data.results.length} produto(s) encontrado(s)`
              : "Nenhum produto encontrado no catálogo."}
          </h2>
          {data.results.length > 0 && (
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-4">
              {data.results.map((product) => (
                <ProductCard key={product.product_id} product={product} />
              ))}
            </div>
          )}
        </section>
      )}
    </main>
  )
}
