"use client"

import { useCallback } from "react"
import { useDropzone } from "react-dropzone"
import { Upload, X } from "lucide-react"
import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import { CameraCapture } from "@/components/CameraCapture"

interface ImageDropzoneProps {
  value: File | null
  onChange: (file: File | null) => void
  className?: string
  disabled?: boolean
}

export function ImageDropzone({ value, onChange, className, disabled = false }: ImageDropzoneProps) {
  const onDrop = useCallback(
    (files: File[]) => { if (files[0] && !disabled) onChange(files[0]) },
    [onChange, disabled],
  )

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "image/*": [] },
    maxFiles: 1,
    noClick: false,
    disabled,
  })

  if (value) {
    return (
      <div className={cn("relative rounded-lg overflow-hidden bg-muted", className)}>
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={URL.createObjectURL(value)}
          alt="Preview"
          className="w-full h-full object-contain"
        />
        {!disabled && (
          <Button
            type="button"
            variant="destructive"
            size="icon"
            className="absolute top-2 right-2 h-7 w-7"
            onClick={() => onChange(null)}
          >
            <X className="h-3.5 w-3.5" />
          </Button>
        )}
      </div>
    )
  }

  return (
    <div className={cn("flex flex-col gap-2", className)}>
      {/* Área de drag-and-drop */}
      <div
        {...getRootProps()}
        className={cn(
          "flex-1 border-2 border-dashed rounded-lg flex flex-col items-center justify-center gap-3 transition-colors select-none min-h-0",
          disabled
            ? "opacity-50 cursor-not-allowed border-muted-foreground/15"
            : isDragActive
              ? "border-primary bg-primary/5 cursor-copy"
              : "border-muted-foreground/25 hover:border-primary/50 hover:bg-muted/40 cursor-pointer",
        )}
      >
        <input {...getInputProps()} />
        <Upload className="h-9 w-9 text-muted-foreground" />
        <div className="text-center px-4">
          <p className="text-sm font-medium">
            {isDragActive ? "Solte a imagem aqui" : "Arraste ou clique para selecionar"}
          </p>
          <p className="text-xs text-muted-foreground mt-1">JPEG, PNG, WebP, BMP</p>
        </div>
      </div>

      {/* Separador */}
      <div className="flex items-center gap-2">
        <div className="flex-1 h-px bg-border" />
        <span className="text-xs text-muted-foreground">ou</span>
        <div className="flex-1 h-px bg-border" />
      </div>

      {/* Botão de câmera */}
      {!disabled && (
        <div className="flex justify-center">
          <CameraCapture onCapture={onChange} />
        </div>
      )}
    </div>
  )
}
