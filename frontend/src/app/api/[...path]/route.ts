/**
 * Proxy reverso server-side para o backend FastAPI.
 *
 * Injeta o X-API-Key a partir de process.env.API_KEY (variável de runtime,
 * não precisa ser baked no bundle do cliente como NEXT_PUBLIC_*).
 * O cliente chama /api/* sem nenhum header de autenticação.
 */
import { type NextRequest, NextResponse } from "next/server"

const BACKEND_URL =
  process.env.API_URL ?? process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:5000"

const API_KEY = process.env.API_KEY ?? ""

async function proxy(request: NextRequest, path: string): Promise<NextResponse> {
  const url = `${BACKEND_URL}/${path}${request.nextUrl.search}`

  const headers = new Headers()
  headers.set("X-API-Key", API_KEY)

  // Repassa Content-Type para uploads multipart/form-data e application/json
  const contentType = request.headers.get("content-type")
  if (contentType) headers.set("content-type", contentType)

  const init: RequestInit = {
    method: request.method,
    headers,
    // body é null para GET/HEAD
    body: ["GET", "HEAD"].includes(request.method) ? null : request.body,
    // @ts-expect-error — duplex é necessário para streaming de body no Node
    duplex: "half",
  }

  try {
    const response = await fetch(url, init)
    const resHeaders = new Headers(response.headers)
    // Remove encoding para evitar erros de descompressão no Next.js
    resHeaders.delete("content-encoding")
    resHeaders.delete("transfer-encoding")

    return new NextResponse(response.body, {
      status: response.status,
      headers: resHeaders,
    })
  } catch (err) {
    console.error("[proxy] Erro ao contatar o backend:", err)
    return NextResponse.json({ detail: "Backend indisponível." }, { status: 502 })
  }
}

export async function GET(request: NextRequest, { params }: { params: Promise<{ path: string[] }> }) {
  const { path } = await params
  return proxy(request, path.join("/"))
}

export async function POST(request: NextRequest, { params }: { params: Promise<{ path: string[] }> }) {
  const { path } = await params
  return proxy(request, path.join("/"))
}

export async function PUT(request: NextRequest, { params }: { params: Promise<{ path: string[] }> }) {
  const { path } = await params
  return proxy(request, path.join("/"))
}

export async function DELETE(request: NextRequest, { params }: { params: Promise<{ path: string[] }> }) {
  const { path } = await params
  return proxy(request, path.join("/"))
}
