'use client'

import { useState, useRef, useCallback } from 'react'
import { saveCVAnalysis, CVAnalysisData } from '@/lib/firebase'

interface CameraCaptureProps {
  onImageCapture?: (imageDataUrl: string, analysisResults?: any) => void
}

export default function CameraCapture({ onImageCapture }: CameraCaptureProps) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [isStreaming, setIsStreaming] = useState(false)
  const [capturedImage, setCapturedImage] = useState<string | null>(null)
  const [uploadedImage, setUploadedImage] = useState<string | null>(null)
  const [error, setError] = useState<string>('')
  const [isLoading, setIsLoading] = useState(false)

  const startCamera = useCallback(async () => {
    try {
      setError('')
      setIsLoading(true)
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { width: 1280, height: 720 } 
      })
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        videoRef.current.play()
        setIsStreaming(true)
      }
    } catch (err) {
      console.error('Error accessing camera:', err)
      setError('Unable to access camera. Please ensure camera permissions are granted.')
    } finally {
      setIsLoading(false)
    }
  }, [])

  const stopCamera = useCallback(() => {
    if (videoRef.current && videoRef.current.srcObject) {
      const tracks = (videoRef.current.srcObject as MediaStream).getTracks()
      tracks.forEach(track => track.stop())
      videoRef.current.srcObject = null
      setIsStreaming(false)
    }
  }, [])

  const captureImage = useCallback(async () => {
    if (!videoRef.current || !canvasRef.current) {
      setError('Camera not ready. Please try again.')
      return
    }

    const video = videoRef.current
    const canvas = canvasRef.current
    const context = canvas.getContext('2d')

    if (!context) {
      setError('Unable to get canvas context.')
      return
    }

    canvas.width = video.videoWidth
    canvas.height = video.videoHeight
    context.drawImage(video, 0, 0, canvas.width, canvas.height)

    const imageDataUrl = canvas.toDataURL('image/jpeg', 0.9)
    setCapturedImage(imageDataUrl)
    
    // Send to computer vision backend for analysis
    try {
      setIsLoading(true)
      const analysisResults = await analyzeImageWithCV(imageDataUrl)
      console.log('CV Analysis Results:', analysisResults)
      
      // Save to Firebase
      await saveAnalysisToFirebase(imageDataUrl, analysisResults)
      
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

  // Helper function to save analysis data to Firebase
  const saveAnalysisToFirebase = useCallback(async (imageDataUrl: string, analysisResults: any, file?: File) => {
    try {
      if (!analysisResults) return

      // Calculate image size
      const imageSize = file?.size || Math.round((imageDataUrl.length * 3) / 4) // Estimate base64 size
      
      // Transform CV results to Firebase format
      const firebaseData = {
        imageMetadata: {
          size: imageSize,
          type: file?.type || 'image/jpeg',
          ...(analysisResults.image_info?.dimensions && { dimensions: analysisResults.image_info.dimensions })
        },
        processingInfo: {
          processingTime: analysisResults.processing_info?.processing_time_ms || 0,
          modelVersion: analysisResults.processing_info?.model_version || 'Smart CV v1.0',
          confidence: analysisResults.processing_info?.overall_confidence || 0.8
        },
        detectionSummary: {
          peopleDetected: analysisResults.detection_summary?.people_detected || 0,
          facesAnalyzed: analysisResults.detection_summary?.faces_analyzed || 0,
          averageConfidence: analysisResults.detection_summary?.average_confidence || 0
        },
        peopleAnalysis: (analysisResults.people || []).map((person: any, index: number) => ({
          personId: index + 1,
          demographics: {
            estimatedAge: person.demographics?.estimated_age || 25,
            ageRange: person.demographics?.age_range || 'adult',
            gender: person.demographics?.gender || 'unknown',
            confidence: person.demographics?.confidence || 0.7
          },
          physicalAttributes: {
            skinTone: person.physical_attributes?.skin_tone || 'medium',
            hairColor: person.physical_attributes?.hair_color || 'brown',
            hairStyle: person.physical_attributes?.hair_style || 'short',
            eyeColor: person.physical_attributes?.eye_color || 'brown'
          },
          appearance: {
            clothing: {
              detected_items: person.appearance?.clothing?.detected_items || [],
              dominant_colors: person.appearance?.clothing?.dominant_colors || [],
              style_category: person.appearance?.clothing?.style_category || 'casual',
              patterns: person.appearance?.clothing?.patterns || [],
              fabric_type: person.appearance?.clothing?.fabric_type || 'cotton'
            },
            accessories: person.appearance?.accessories || [],
            overall_style: person.appearance?.overall_style || 'casual',
            outfit_formality: person.appearance?.outfit_formality || 'casual'
          },
          emotions: {
            primary: person.emotions?.primary || 'neutral',
            confidence: person.emotions?.confidence || 0.6,
            ...(person.emotions?.secondary && { secondary: person.emotions.secondary })
          },
          pose: {
            position: person.pose?.position || 'standing',
            orientation: person.pose?.orientation || 'front',
            visibility: person.pose?.visibility || 'full'
          }
        })),
        sceneAnalysis: {
          lighting: analysisResults.scene_analysis?.lighting || 'natural',
          setting: analysisResults.scene_analysis?.setting || 'indoor',
          imageQuality: analysisResults.scene_analysis?.image_quality || 'good',
          dominantColors: analysisResults.scene_analysis?.dominant_colors || []
        }
      }

      const docId = await saveCVAnalysis(firebaseData)
      console.log('Analysis saved to Firebase with ID:', docId)
      
    } catch (error) {
      console.error('Failed to save analysis to Firebase:', error)
      // Don't throw error to avoid breaking the main flow
    }
  }, [])

  // Computer vision analysis function
  const analyzeImageWithCV = useCallback(async (imageDataUrl: string) => {
    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5000'
      console.log('🔍 DEBUG: API URL being used:', apiUrl)
      console.log('🔍 DEBUG: Environment variable NEXT_PUBLIC_API_URL:', process.env.NEXT_PUBLIC_API_URL)
      
      const response = await fetch(`${apiUrl}/analyze-image`, {
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

      const reader = new FileReader()
      reader.onload = async (e) => {
        const imageDataUrl = e.target?.result as string
        setUploadedImage(imageDataUrl)
        setCapturedImage(imageDataUrl) // Use the same state for display

        // Analyze the uploaded image
        try {
                 const analysisResults = await analyzeImageWithCV(imageDataUrl)
                 console.log('CV Analysis Results:', analysisResults)
                 
                 // Save to Firebase
                 await saveAnalysisToFirebase(imageDataUrl, analysisResults, file)
                 
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
    link.download = `captured-image-${Date.now()}.jpg`
    link.href = capturedImage
    link.click()
  }, [capturedImage])

  return (
    <div className="space-y-6">
      {/* Error Message */}
      {error && (
        <div className="p-4 bg-red-50 border border-red-200 rounded-lg">
          <div className="flex items-center gap-2">
            <span className="text-red-600">⚠️</span>
            <span className="text-red-700">{error}</span>
          </div>
        </div>
      )}

      {/* Camera Preview or Image Display */}
      <div className="relative">
        {!capturedImage ? (
          <>
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className={`w-full rounded-lg shadow-lg ${
                isStreaming ? 'block' : 'hidden'
              }`}
              style={{ maxHeight: '400px', objectFit: 'cover' }}
            />
            
            {!isStreaming && (
              <div className="w-full h-64 bg-gray-100 rounded-lg flex items-center justify-center">
                <div className="text-center">
                  <div className="w-16 h-16 bg-gray-200 rounded-lg flex items-center justify-center mx-auto mb-4">
                    <svg className="w-8 h-8 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z" />
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 13a3 3 0 11-6 0 3 3 0 016 0z" />
                    </svg>
                  </div>
                  <p className="text-gray-500">Camera preview will appear here</p>
                </div>
              </div>
            )}
          </>
        ) : (
          <div className="relative">
            <img
              src={capturedImage}
              alt="Captured"
              className="w-full rounded-lg shadow-lg"
              style={{ maxHeight: '400px', objectFit: 'cover' }}
            />
            <div className="absolute top-4 left-4">
              <span className="corporate-badge corporate-badge-success">
                {uploadedImage ? 'Uploaded Image' : 'Captured Image'}
              </span>
            </div>
          </div>
        )}
        
        <canvas ref={canvasRef} className="hidden" />
      </div>

      {/* Control Buttons */}
      <div className="space-y-3">
        {!isStreaming && !capturedImage ? (
          <>
            <button
              onClick={startCamera}
              disabled={isLoading}
              className="btn-corporate-primary w-full"
            >
              {isLoading ? (
                <div className="flex items-center gap-2">
                  <div className="corporate-spinner"></div>
                  Starting...
                </div>
              ) : (
                <div className="flex items-center gap-2">
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                  </svg>
                  Start Camera
                </div>
              )}
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
                className={`btn-corporate-secondary w-full flex items-center justify-center gap-2 ${
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
        ) : isStreaming ? (
          <div className="flex gap-3">
            <button
              onClick={captureImage}
              disabled={isLoading}
              className="btn-corporate-primary flex-1"
            >
              {isLoading ? (
                <div className="flex items-center gap-2">
                  <div className="corporate-spinner"></div>
                  Processing...
                </div>
              ) : (
                <div className="flex items-center gap-2">
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z" />
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 13a3 3 0 11-6 0 3 3 0 016 0z" />
                  </svg>
                  Capture Image
                </div>
              )}
            </button>
            <button
              onClick={stopCamera}
              className="btn-corporate-secondary"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 10a1 1 0 011-1h4a1 1 0 011 1v4a1 1 0 01-1 1h-4a1 1 0 01-1-1v-4z" />
              </svg>
            </button>
          </div>
        ) : (
          <div className="flex gap-3">
            <button
              onClick={retakePhoto}
              className="btn-corporate-secondary flex-1"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
              </svg>
              {uploadedImage ? 'Clear Image' : 'Retake Photo'}
            </button>
            <button
              onClick={downloadImage}
              className="btn-corporate-primary"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
              Download
            </button>
          </div>
        )}
      </div>
    </div>
  )
} 