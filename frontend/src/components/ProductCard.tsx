"use client"

import { useState } from "react"
import { Card, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Skeleton } from "@/components/ui/skeleton"
import type { SearchResult } from "@/types/api"
import { formatSimilarity } from "@/utils/format"

const TYPE_LABELS: Record<string, string> = {
  clean: "Fundo Branco",
  environment: "Ambiente",
  person: "Pessoa",
}

interface ProductCardProps {
  product: SearchResult
}

export function ProductCard({ product }: ProductCardProps) {
  const [activeImage, setActiveImage] = useState(product.main_image)
  const similarity = product.similarity
  const variant = similarity >= 0.8 ? "default" : similarity >= 0.5 ? "secondary" : "outline"

  // Monta lista de thumbnails: todas as imagens de todos os tipos
  const thumbnails = Object.entries(product.images).flatMap(([type, urls]) =>
    urls.map((url) => ({ url, type }))
  )

  return (
    <Card className="overflow-hidden group">
      {/* Imagem principal */}
      <div className="aspect-square bg-muted overflow-hidden">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={activeImage}
          alt={product.name}
          className="w-full h-full object-cover transition-transform duration-300 group-hover:scale-105"
        />
      </div>

      <CardContent className="p-3 space-y-2">
        <div className="flex items-start justify-between gap-1">
          <p className="text-sm font-medium truncate" title={product.name}>
            {product.name}
          </p>
          <Badge variant={variant} className="text-xs shrink-0">
            {formatSimilarity(similarity)}
          </Badge>
        </div>

        {/* Thumbnails das variantes */}
        {thumbnails.length > 1 && (
          <div className="flex gap-1 flex-wrap">
            {thumbnails.map(({ url, type }) => (
              <button
                key={url}
                title={TYPE_LABELS[type] ?? type}
                onClick={() => setActiveImage(url)}
                className={`w-10 h-10 rounded overflow-hidden border-2 transition-colors ${
                  activeImage === url ? "border-primary" : "border-transparent"
                }`}
              >
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={url} alt={type} className="w-full h-full object-cover" />
              </button>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  )
}

export function ProductCardSkeleton() {
  return (
    <Card className="overflow-hidden">
      <Skeleton className="aspect-square w-full rounded-none" />
      <CardContent className="p-3 space-y-2">
        <div className="flex justify-between">
          <Skeleton className="h-4 w-3/4" />
          <Skeleton className="h-5 w-12 rounded-full" />
        </div>
        <div className="flex gap-1">
          <Skeleton className="h-10 w-10 rounded" />
          <Skeleton className="h-10 w-10 rounded" />
        </div>
      </CardContent>
    </Card>
  )
}
