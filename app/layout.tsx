import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'
import CorporateHeader from '@/components/CorporateHeader'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'Computer Vision Analytics',
  description: 'Advanced computer vision analysis for clothing, accessories, and personal attributes',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <div className="min-h-screen bg-gray-50">
          <CorporateHeader />
          <main className="corporate-container">
            {children}
          </main>
        </div>
      </body>
    </html>
  )
} 