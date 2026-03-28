"use client"

import { useState } from "react"
import { useForm } from "react-hook-form"
import { zodResolver } from "@hookform/resolvers/zod"
import { z } from "zod"
import Link from "next/link"
import { ArrowLeft, CheckCircle2, Loader2, PackagePlus } from "lucide-react"
import { toast } from "sonner"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Form, FormControl, FormField, FormItem, FormLabel, FormMessage } from "@/components/ui/form"
import { ImageDropzone } from "@/components/ImageDropzone"
import { useSaveProductImage } from "@/queries/productQueries"
import type { ImageType } from "@/types/api"

// Slots das 3 imagens — ordem define a sequência de upload
const IMAGE_SLOTS: { type: ImageType; label: string; hint: string; required: boolean }[] = [
  { type: "clean",       label: "Fundo Branco", hint: "Gera o embedding — base da busca", required: true  },
  { type: "environment", label: "Ambiente",     hint: "Produto em contexto (opcional)",    required: false },
  { type: "person",      label: "Pessoa",       hint: "Modelo usando o produto (opcional)", required: false },
]

const schema = z.object({
  productId: z
    .string()
    .min(1, "ID do produto é obrigatório")
    .regex(/^\S+$/, "ID não pode conter espaços"),
})
type FormValues = z.infer<typeof schema>

type FileMap = Record<ImageType, File | null>
type UrlMap  = Record<ImageType, string | null>

const emptyFiles = (): FileMap => ({ clean: null, environment: null, person: null })
const emptyUrls  = (): UrlMap  => ({ clean: null, environment: null, person: null })

export default function NewProductPage() {
  const [files,    setFiles]    = useState<FileMap>(emptyFiles())
  const [savedUrls, setSavedUrls] = useState<UrlMap>(emptyUrls())
  const [progress, setProgress] = useState<{ label: string; step: number; total: number } | null>(null)

  const { mutateAsync } = useSaveProductImage()
  const isUploading = progress !== null

  const form = useForm<FormValues>({
    resolver: zodResolver(schema),
    defaultValues: { productId: "" },
  })

  function setFile(type: ImageType, file: File | null) {
    setFiles((prev) => ({ ...prev, [type]: file }))
  }

  async function onSubmit({ productId }: FormValues) {
    if (!files.clean) {
      toast.error("A imagem Fundo Branco é obrigatória.")
      return
    }

    // Monta lista de uploads na ordem: clean → environment → person
    const queue = IMAGE_SLOTS
      .filter((slot) => files[slot.type] !== null)
      .map((slot) => ({ type: slot.type, label: slot.label, file: files[slot.type]! }))

    const newUrls: UrlMap = emptyUrls()
    let allOk = true

    for (let i = 0; i < queue.length; i++) {
      const { type, label, file } = queue[i]
      setProgress({ label, step: i + 1, total: queue.length })

      try {
        const result = await mutateAsync({ file, productId, imageType: type })
        newUrls[type] = result.url
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : "Erro desconhecido"
        toast.error(`Erro ao enviar ${label}: ${msg}`)
        allOk = false
        break
      }
    }

    setProgress(null)

    if (allOk) {
      toast.success(`Produto ${productId} salvo com ${queue.length} imagem(ns).`)
      setSavedUrls(newUrls)
      setFiles(emptyFiles())
      form.reset()
    }
  }

  const hasAnyFile = Object.values(files).some(Boolean)

  return (
    <main className="container max-w-3xl mx-auto py-12 px-4">
      <Link
        href="/"
        className="inline-flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground transition-colors mb-6"
      >
        <ArrowLeft className="h-4 w-4" />
        Voltar para busca
      </Link>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <PackagePlus className="h-5 w-5" />
            Adicionar Produto
          </CardTitle>
        </CardHeader>

        <CardContent>
          <Form {...form}>
            <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-6">

              {/* ID do produto */}
              <FormField
                control={form.control}
                name="productId"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>ID do Produto</FormLabel>
                    <FormControl>
                      <Input placeholder="Ex: 11666008" disabled={isUploading} {...field} />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />

              {/* Três dropzones em grid */}
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                {IMAGE_SLOTS.map((slot) => (
                  <div key={slot.type} className="space-y-2">
                    <div className="flex items-center gap-2 flex-wrap">
                      <span className="text-sm font-medium">{slot.label}</span>
                      {slot.required
                        ? <Badge variant="destructive" className="text-[10px] px-1.5 py-0">Obrigatório</Badge>
                        : <Badge variant="outline"     className="text-[10px] px-1.5 py-0">Opcional</Badge>
                      }
                    </div>
                    <p className="text-[11px] text-muted-foreground">{slot.hint}</p>
                    <ImageDropzone
                      value={files[slot.type]}
                      onChange={(f) => setFile(slot.type, f)}
                      className="h-44"
                      disabled={isUploading}
                    />
                  </div>
                ))}
              </div>

              {/* Barra de progresso durante o upload */}
              {progress && (
                <div className="rounded-lg border bg-muted/40 px-4 py-3 space-y-1.5">
                  <div className="flex items-center gap-2 text-sm font-medium">
                    <Loader2 className="h-4 w-4 animate-spin text-primary" />
                    Enviando {progress.label}… ({progress.step}/{progress.total})
                  </div>
                  <div className="w-full bg-border rounded-full h-1.5">
                    <div
                      className="bg-primary h-1.5 rounded-full transition-all duration-500"
                      style={{ width: `${(progress.step / progress.total) * 100}%` }}
                    />
                  </div>
                </div>
              )}

              <Button
                type="submit"
                className="w-full"
                disabled={!hasAnyFile || isUploading}
              >
                {isUploading ? (
                  <><Loader2 className="mr-2 h-4 w-4 animate-spin" /> Enviando...</>
                ) : (
                  "Salvar Produto"
                )}
              </Button>
            </form>
          </Form>

          {/* Previews das imagens salvas */}
          {Object.values(savedUrls).some(Boolean) && (
            <div className="mt-6 border-t pt-5 space-y-3">
              <p className="text-sm font-medium text-muted-foreground flex items-center gap-1.5">
                <CheckCircle2 className="h-4 w-4 text-green-500" />
                Produto salvo com sucesso
              </p>
              <div className="grid grid-cols-3 gap-3">
                {IMAGE_SLOTS.map((slot) =>
                  savedUrls[slot.type] ? (
                    <div key={slot.type} className="space-y-1">
                      <p className="text-[11px] text-muted-foreground text-center">{slot.label}</p>
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img
                        src={savedUrls[slot.type]!}
                        alt={slot.label}
                        className="w-full rounded-lg object-contain aspect-square bg-muted"
                      />
                    </div>
                  ) : null
                )}
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </main>
  )
}
