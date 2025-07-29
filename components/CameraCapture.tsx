'use client'

import { useState, useRef, useCallback } from 'react'

interface CameraCaptureProps {
  onImageCapture?: (imageDataUrl: string, analysisResults?: any) => void
}

export default function CameraCapture({ onImageCapture }: CameraCaptureProps) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [isStreaming, setIsStreaming] = useState(false)
  const [capturedImage, setCapturedImage] = useState<string | null>(null)
  const [error, setError] = useState<string>('')
  const [isLoading, setIsLoading] = useState(false)
  const [uploadedImage, setUploadedImage] = useState<string | null>(null)

  const startCamera = useCallback(async () => {
    setIsLoading(true)
    setError('')
    
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: 'user'
        }
      })
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        setIsStreaming(true)
      }
    } catch (err) {
      console.error('Error accessing camera:', err)
      setError('Unable to access camera. Please ensure you have granted camera permissions.')
    } finally {
      setIsLoading(false)
    }
  }, [])

  const stopCamera = useCallback(() => {
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream
      stream.getTracks().forEach(track => track.stop())
      videoRef.current.srcObject = null
      setIsStreaming(false)
    }
  }, [])

  const captureImage = useCallback(async () => {
    if (!videoRef.current || !canvasRef.current) return

    const video = videoRef.current
    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')

    if (!ctx) return

    // Set canvas dimensions to match video
    canvas.width = video.videoWidth
    canvas.height = video.videoHeight

    // Draw the video frame to the canvas
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height)

    // Get the image data as base64
    const imageDataUrl = canvas.toDataURL('image/jpeg', 0.9)
    setCapturedImage(imageDataUrl)
    
    // Send to computer vision backend for analysis
    try {
      setIsLoading(true)
      const analysisResults = await analyzeImageWithCV(imageDataUrl)
      console.log('CV Analysis Results:', analysisResults)
      
      if (onImageCapture) {
        onImageCapture(imageDataUrl, analysisResults)
      }
    } catch (error) {
      console.error('Error analyzing image:', error)
      setError('Failed to analyze image. Please try again.')
      
      if (onImageCapture) {
        onImageCapture(imageDataUrl)
      }
    } finally {
      setIsLoading(false)
    }
  }, [onImageCapture])

  const retakePhoto = useCallback(() => {
    setCapturedImage(null)
    setUploadedImage(null)
  }, [])

  // Computer vision analysis function
  const analyzeImageWithCV = useCallback(async (imageDataUrl: string) => {
    try {
      const response = await fetch('http://localhost:5000/analyze-image', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image: imageDataUrl
        })
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const results = await response.json()
      return results
    } catch (error) {
      console.error('Computer vision analysis failed:', error)
      throw error
    }
  }, [])

  const handleImageUpload = useCallback(async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return

    // Validate file type
    if (!file.type.startsWith('image/')) {
      setError('Please select a valid image file (JPEG, PNG, WebP)')
      return
    }

    // Validate file size (max 10MB)
    if (file.size > 10 * 1024 * 1024) {
      setError('Image file too large. Please select an image under 10MB.')
      return
    }

    try {
      setIsLoading(true)
      setError('')

      // Convert file to base64
      const reader = new FileReader()
      reader.onload = async (e) => {
        const imageDataUrl = e.target?.result as string
        setUploadedImage(imageDataUrl)
        setCapturedImage(imageDataUrl) // Use the same state for display

        // Analyze the uploaded image
        try {
          const analysisResults = await analyzeImageWithCV(imageDataUrl)
          console.log('CV Analysis Results:', analysisResults)
          
          if (onImageCapture) {
            onImageCapture(imageDataUrl, analysisResults)
          }
        } catch (error) {
          console.error('Error analyzing uploaded image:', error)
          setError('Failed to analyze image. Please try again.')
          
          if (onImageCapture) {
            onImageCapture(imageDataUrl)
          }
        } finally {
          setIsLoading(false)
        }
      }

      reader.onerror = () => {
        setError('Failed to read the image file. Please try again.')
        setIsLoading(false)
      }

      reader.readAsDataURL(file)
    } catch (error) {
      console.error('Error uploading image:', error)
      setError('Failed to upload image. Please try again.')
      setIsLoading(false)
    }
  }, [onImageCapture])

  const downloadImage = useCallback(() => {
    if (!capturedImage) return

    const link = document.createElement('a')
    link.download = `image-${Date.now()}.jpg`
    link.href = capturedImage
    link.click()
  }, [capturedImage])

  return (
    <div className="w-full max-w-4xl mx-auto">
      {error && (
        <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg">
          <p className="text-red-700">{error}</p>
        </div>
      )}

      <div className="grid lg:grid-cols-2 gap-8">
        {/* Camera Preview Section */}
        <div className="space-y-4">
          <h3 className="text-xl font-semibold text-gray-900">Camera Preview</h3>
          
          <div className="relative bg-gray-100 rounded-lg overflow-hidden" style={{ aspectRatio: '16/9' }}>
            {!isStreaming && !isLoading && (
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="text-center">
                  <svg className="w-16 h-16 text-gray-400 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z" />
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 13a3 3 0 11-6 0 3 3 0 016 0z" />
                  </svg>
                  <p className="text-gray-500">Camera preview will appear here</p>
                </div>
              </div>
            )}
            
            {isLoading && (
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="text-center">
                  <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
                  <p className="text-gray-500">Starting camera...</p>
                </div>
              </div>
            )}
            
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className={`w-full h-full object-cover camera-preview ${!isStreaming ? 'hidden' : ''}`}
            />
          </div>

          <div className="space-y-3">
            {!isStreaming ? (
              <>
                <button
                  onClick={startCamera}
                  disabled={isLoading}
                  className="btn-primary w-full"
                >
                  {isLoading ? 'Starting...' : 'Start Camera'}
                </button>
                
                <div className="relative">
                  <input
                    type="file"
                    accept="image/*"
                    onChange={handleImageUpload}
                    disabled={isLoading}
                    className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                    id="image-upload"
                  />
                  <label
                    htmlFor="image-upload"
                    className={`btn-secondary w-full flex items-center justify-center gap-2 ${
                      isLoading ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'
                    }`}
                  >
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                    </svg>
                    {isLoading ? 'Processing...' : 'Upload Image'}
                  </label>
                </div>
              </>
            ) : (
              <div className="flex gap-3">
                <button
                  onClick={captureImage}
                  disabled={isLoading}
                  className="btn-primary flex-1"
                >
                  {isLoading ? 'Processing...' : 'Capture Image'}
                </button>
                <button
                  onClick={stopCamera}
                  className="btn-secondary"
                >
                  Stop Camera
                </button>
              </div>
            )}
          </div>
        </div>

        {/* Captured Image Section */}
                  <div className="space-y-4">
            <h3 className="text-xl font-semibold text-gray-900">
              {uploadedImage ? 'Uploaded Image' : 'Captured Image'}
            </h3>
          
          <div className="relative bg-gray-100 rounded-lg overflow-hidden" style={{ aspectRatio: '16/9' }}>
            {capturedImage ? (
              <img
                src={capturedImage}
                alt="Captured"
                className="w-full h-full object-cover captured-image"
              />
            ) : (
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="text-center">
                  <svg className="w-16 h-16 text-gray-400 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                  </svg>
                  <p className="text-gray-500">
                    {uploadedImage ? 'Uploaded image will appear here' : 'Captured or uploaded image will appear here'}
                  </p>
                </div>
              </div>
            )}
          </div>

          {capturedImage && (
            <div className="flex gap-3">
              <button
                onClick={retakePhoto}
                className="btn-secondary flex-1"
              >
                {uploadedImage ? 'Clear Image' : 'Retake Photo'}
              </button>
              <button
                onClick={downloadImage}
                className="btn-primary"
              >
                Download
              </button>
            </div>
          )}
        </div>
      </div>

      {/* Hidden canvas for image capture */}
      <canvas ref={canvasRef} className="hidden" />
    </div>
  )
} 