/** @type {import('next').NextConfig} */
const nextConfig = {
  // Azure Static Web Apps configuration
  output: 'export',
  trailingSlash: true,
  skipTrailingSlashRedirect: true,
  distDir: 'out',
  images: {
    unoptimized: true
  },
  
  // Optimize for production builds
  compress: true,
  
  // Reduce memory usage during build
  experimental: {
    workerThreads: false,
    cpus: 1
  },
  
  // Disable source maps in production to reduce build size
  productionBrowserSourceMaps: false,
  
  // Optimize bundle analyzer
  webpack: (config, { isServer }) => {
    // Reduce memory usage
    config.optimization = {
      ...config.optimization,
      splitChunks: {
        chunks: 'all',
        maxSize: 244000, // Limit chunk size to prevent memory issues
        cacheGroups: {
          vendor: {
            test: /[\\/]node_modules[\\/]/,
            name: 'vendors',
            chunks: 'all',
          },
        },
      },
    }
    
    return config
  },
}

module.exports = nextConfig 