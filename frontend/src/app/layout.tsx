import type { Metadata } from "next"
import { Inter } from "next/font/google"
import Link from "next/link"
import { Gem, PlusCircle } from "lucide-react"
import "./globals.css"
import { Providers } from "./providers"

const inter = Inter({ subsets: ["latin"] })

export const metadata: Metadata = {
  title: "GemFinder",
  description: "Busca de joias por similaridade visual",
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="pt-BR" suppressHydrationWarning>
      <body className={inter.className}>
        <Providers>
          <header className="sticky top-0 z-10 border-b bg-background/80 backdrop-blur">
            <div className="container max-w-5xl mx-auto px-4 h-14 flex items-center justify-between">
              <Link href="/" className="flex items-center gap-2 font-bold text-primary">
                <Gem className="h-5 w-5" />
                GemFinder
              </Link>
              <Link
                href="/products/new"
                className="flex items-center gap-1.5 text-sm text-muted-foreground hover:text-foreground transition-colors"
              >
                <PlusCircle className="h-4 w-4" />
                Adicionar Produto
              </Link>
            </div>
          </header>
          {children}
        </Providers>
      </body>
    </html>
  )
}
