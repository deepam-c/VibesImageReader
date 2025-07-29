export default function HomePage() {
  return (
    <div className="min-h-screen p-6 lg:p-8">
      <div className="max-w-4xl mx-auto">
        <div className="text-center mb-12">
          <h1 className="text-4xl lg:text-6xl font-bold text-gray-900 mb-6">
            Welcome to{' '}
            <span className="bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
              Camera Capture
            </span>
          </h1>
          <p className="text-xl text-gray-600 max-w-2xl mx-auto">
            A professional and elegant web application for capturing high-quality images 
            from your device camera. Works seamlessly on both desktop and mobile devices.
          </p>
        </div>

        <div className="grid md:grid-cols-2 gap-8 mb-12">
          <div className="elegant-card p-8">
            <div className="w-16 h-16 bg-gradient-to-br from-blue-500 to-purple-600 rounded-xl flex items-center justify-center mb-6">
              <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 13a3 3 0 11-6 0 3 3 0 016 0z" />
              </svg>
            </div>
            <h3 className="text-2xl font-semibold text-gray-900 mb-4">High-Quality Capture</h3>
            <p className="text-gray-600">
              Capture crisp, clear images using your device's camera with our advanced 
              web-based capture technology.
            </p>
          </div>

          <div className="elegant-card p-8">
            <div className="w-16 h-16 bg-gradient-to-br from-green-500 to-blue-600 rounded-xl flex items-center justify-center mb-6">
              <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 18h.01M8 21h8a2 2 0 002-2V5a2 2 0 00-2-2H8a2 2 0 00-2 2v14a2 2 0 002 2z" />
              </svg>
            </div>
            <h3 className="text-2xl font-semibold text-gray-900 mb-4">Cross-Device Support</h3>
            <p className="text-gray-600">
              Works flawlessly on laptops, tablets, and mobile phones. Responsive design 
              ensures optimal experience on any screen size.
            </p>
          </div>
        </div>

        <div className="text-center">
          <div className="elegant-card p-8 inline-block">
            <h2 className="text-2xl font-semibold text-gray-900 mb-4">Ready to get started?</h2>
            <p className="text-gray-600 mb-6">
              Use the sidebar navigation to access the camera capture feature
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <a 
                href="/capture" 
                className="btn-primary inline-block text-center"
              >
                Start Capturing
              </a>
              <button className="btn-secondary">
                Learn More
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
} 