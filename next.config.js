/** @type {import('next').NextConfig} */
const nextConfig = {
  // App Router is now stable in Next.js 15.4.4, no experimental config needed
  output: 'export',
  trailingSlash: true,
  skipTrailingSlashRedirect: true,
  distDir: 'out',
  images: {
    unoptimized: true
  }
}

module.exports = nextConfig 