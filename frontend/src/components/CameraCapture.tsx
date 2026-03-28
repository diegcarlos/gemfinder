"use client"

import { useCallback, useEffect, useRef, useState } from "react"
import { Camera, FlipHorizontal, X } from "lucide-react"
import { Button } from "@/components/ui/button"

interface CameraCaptureProps {
  onCapture: (file: File) => void
}

export function CameraCapture({ onCapture }: CameraCaptureProps) {
  const [isOpen, setIsOpen] = useState(false)
  const [stream, setStream] = useState<MediaStream | null>(null)
  const [facingMode, setFacingMode] = useState<"environment" | "user">("environment")
  const [error, setError] = useState<string | null>(null)
  const videoRef = useRef<HTMLVideoElement>(null)

  // Anexa o stream ao elemento <video> assim que ambos estiverem prontos
  useEffect(() => {
    if (videoRef.current && stream) {
      videoRef.current.srcObject = stream
    }
  }, [stream, isOpen])

  // Garante que o stream seja liberado ao desmontar o componente
  useEffect(() => {
    return () => stopStream(stream)
  }, [stream])

  function stopStream(s: MediaStream | null) {
    s?.getTracks().forEach((t) => t.stop())
  }

  const openCamera = useCallback(async (facing: "environment" | "user" = facingMode) => {
    setError(null)
    stopStream(stream)

    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: facing, width: { ideal: 1280 }, height: { ideal: 720 } },
      })
      setStream(mediaStream)
      setIsOpen(true)
    } catch {
      setError("Não foi possível acessar a câmera. Verifique as permissões do navegador.")
    }
  }, [stream, facingMode])

  const closeCamera = useCallback(() => {
    stopStream(stream)
    setStream(null)
    setIsOpen(false)
    setError(null)
  }, [stream])

  const flipCamera = useCallback(() => {
    const next = facingMode === "environment" ? "user" : "environment"
    setFacingMode(next)
    openCamera(next)
  }, [facingMode, openCamera])

  const capture = useCallback(() => {
    const video = videoRef.current
    if (!video) return

    const canvas = document.createElement("canvas")
    canvas.width = video.videoWidth
    canvas.height = video.videoHeight
    canvas.getContext("2d")?.drawImage(video, 0, 0)

    canvas.toBlob(
      (blob) => {
        if (!blob) return
        const file = new File([blob], `foto_${Date.now()}.jpg`, { type: "image/jpeg" })
        onCapture(file)
        closeCamera()
      },
      "image/jpeg",
      0.92,
    )
  }, [onCapture, closeCamera])

  return (
    <>
      <Button type="button" variant="outline" size="sm" onClick={() => openCamera()}>
        <Camera className="mr-1.5 h-3.5 w-3.5" />
        Tirar Foto
      </Button>

      {error && <p className="text-xs text-destructive mt-1">{error}</p>}

      {/* Overlay full-screen da câmera */}
      {isOpen && (
        <div className="fixed inset-0 z-50 bg-black flex flex-col">
          {/* Barra superior */}
          <div className="flex items-center justify-between px-4 py-3 bg-black/60">
            <span className="text-white text-sm font-medium">Câmera</span>
            <button
              onClick={closeCamera}
              className="text-white p-1 rounded-full hover:bg-white/20 transition-colors"
              aria-label="Fechar câmera"
            >
              <X className="h-5 w-5" />
            </button>
          </div>

          {/* Stream de vídeo */}
          <div className="flex-1 flex items-center justify-center overflow-hidden">
            {/* eslint-disable-next-line jsx-a11y/media-has-caption */}
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className="max-w-full max-h-full object-contain"
            />
          </div>

          {/* Barra inferior: botão de captura + flip */}
          <div className="flex items-center justify-center gap-8 px-6 py-8 bg-black/60">
            {/* Flip câmera (frente/traseira) */}
            <button
              onClick={flipCamera}
              className="text-white p-3 rounded-full hover:bg-white/20 transition-colors"
              aria-label="Alternar câmera"
            >
              <FlipHorizontal className="h-6 w-6" />
            </button>

            {/* Botão de captura */}
            <button
              onClick={capture}
              className="w-16 h-16 rounded-full bg-white border-4 border-gray-400 hover:scale-105 active:scale-95 transition-transform shadow-lg"
              aria-label="Capturar foto"
            />

            {/* Espaçador para centralizar o botão de captura */}
            <div className="w-12 h-12" />
          </div>
        </div>
      )}
    </>
  )
}
