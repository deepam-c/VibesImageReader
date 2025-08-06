/** @type {import('next').NextConfig} */
const nextConfig = {
  // Azure Static Web Apps - minimal configuration
  output: 'export',
  trailingSlash: true,
  distDir: 'out',
  images: {
    unoptimized: true
  },
  
  // Minimal optimizations to prevent build issues
  swcMinify: false, // Disable SWC minifier that might cause memory issues
  
  // Disable experimental features that might cause problems
  experimental: {},
  
  // Simple webpack config
  webpack: (config) => {
    // Minimal webpack modifications
    config.optimization.minimize = false // Disable minification during build
    return config
  },
}

module.exports = nextConfig 