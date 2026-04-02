"use client";

import { useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import type { SearchResult } from "@/types/api";
import { formatSimilarity } from "@/utils/format";
import { X, ChevronLeft, ChevronRight } from "lucide-react";

const TYPE_LABELS: Record<string, string> = {
  clean: "Fundo Branco",
  environment: "Ambiente",
  person: "Pessoa",
};

interface ProductCardProps {
  product: SearchResult;
}

interface Thumbnail {
  url: string;
  type: string;
}

// ── Modal de visualização ampliada ────────────────────────────────────────────

function ImageModal({
  thumbnails,
  initialIndex,
  productName,
  onClose,
}: {
  thumbnails: Thumbnail[];
  initialIndex: number;
  productName: string;
  onClose: () => void;
}) {
  const [current, setCurrent] = useState(initialIndex);

  const prev = () => setCurrent((i) => (i - 1 + thumbnails.length) % thumbnails.length);
  const next = () => setCurrent((i) => (i + 1) % thumbnails.length);

  const handleBackdropClick = (e: React.MouseEvent) => {
    if (e.target === e.currentTarget) onClose();
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4"
      onClick={handleBackdropClick}
    >
      <div className="relative bg-background rounded-xl shadow-2xl max-w-3xl w-full max-h-[90vh] flex flex-col overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b shrink-0">
          <div className="flex items-center gap-2">
            <p className="font-medium text-sm">{productName}</p>
            <Badge variant="outline" className="text-xs">
              {TYPE_LABELS[thumbnails[current].type] ?? thumbnails[current].type}
            </Badge>
          </div>
          <button
            onClick={onClose}
            className="rounded-full p-1 hover:bg-muted transition-colors"
            aria-label="Fechar"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Imagem principal */}
        <div className="relative flex-1 flex items-center justify-center bg-muted/30 min-h-0">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={thumbnails[current].url}
            alt={productName}
            className="max-w-full max-h-full object-contain p-4"
          />

          {thumbnails.length > 1 && (
            <>
              <button
                onClick={prev}
                className="absolute left-2 p-2 rounded-full bg-background/80 hover:bg-background shadow transition-colors"
                aria-label="Anterior"
              >
                <ChevronLeft className="w-5 h-5" />
              </button>
              <button
                onClick={next}
                className="absolute right-2 p-2 rounded-full bg-background/80 hover:bg-background shadow transition-colors"
                aria-label="Próxima"
              >
                <ChevronRight className="w-5 h-5" />
              </button>
            </>
          )}
        </div>

        {/* Thumbnails */}
        {thumbnails.length > 1 && (
          <div className="flex gap-2 p-3 border-t overflow-x-auto shrink-0 justify-center">
            {thumbnails.map(({ url, type }, i) => (
              <button
                key={url}
                onClick={() => setCurrent(i)}
                title={TYPE_LABELS[type] ?? type}
                className={`w-14 h-14 rounded-lg overflow-hidden border-2 shrink-0 transition-colors ${
                  i === current ? "border-primary" : "border-transparent hover:border-muted-foreground/40"
                }`}
              >
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={url} alt={type} className="w-full h-full object-cover" />
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}


// ── Card do produto ───────────────────────────────────────────────────────────

export function ProductCard({ product }: ProductCardProps) {
  const [activeImage, setActiveImage] = useState(product.main_image);
  const [modalIndex, setModalIndex] = useState<number | null>(null);

  const similarity = product.similarity;
  const variant =
    similarity >= 0.8 ? "default" : similarity >= 0.5 ? "secondary" : "outline";

  const thumbnails: Thumbnail[] = Object.entries(product.images).flatMap(([type, urls]) =>
    (urls as string[]).map((url) => ({ url, type })),
  );

  const activeIndex = thumbnails.findIndex((t) => t.url === activeImage) ?? 0;

  return (
    <>
      <Card className="overflow-hidden group cursor-pointer">
        {/* Imagem principal — clique abre o modal */}
        <div
          className="aspect-square bg-muted overflow-hidden"
          onClick={() => setModalIndex(activeIndex >= 0 ? activeIndex : 0)}
        >
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
              {thumbnails.map(({ url, type }, i) => (
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

      {/* Modal */}
      {modalIndex !== null && (
        <ImageModal
          thumbnails={thumbnails.length > 0 ? thumbnails : [{ url: activeImage, type: "clean" }]}
          initialIndex={modalIndex}
          productName={product.name}
          onClose={() => setModalIndex(null)}
        />
      )}
    </>
  );
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
  );
}
