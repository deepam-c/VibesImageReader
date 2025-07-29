'use client'

import { useState } from 'react'
import CameraCapture from '@/components/CameraCapture'
import AnalysisResults from '@/components/AnalysisResults'

export default function CapturePage() {
  const [analysisResults, setAnalysisResults] = useState<any>(null)
  const [showAnalysis, setShowAnalysis] = useState(false)
  return (
    <div className="min-h-screen p-6 lg:p-8">
      <div className="max-w-6xl mx-auto">
        <div className="mb-8">
          <h1 className="text-3xl lg:text-4xl font-bold text-gray-900 mb-4">
            Camera Capture
          </h1>
          <p className="text-lg text-gray-600">
            Capture high-quality images using your device camera. Make sure to allow camera permissions when prompted.
          </p>
        </div>

        <div className="elegant-card p-6 lg:p-8">
          <CameraCapture
            onImageCapture={(imageDataUrl, analysisResults) => {
              console.log('Image captured:', imageDataUrl.slice(0, 50) + '...')
              if (analysisResults) {
                console.log('Analysis results:', analysisResults)
                setAnalysisResults(analysisResults)
                setShowAnalysis(true)
              }
            }}
          />
          
          <AnalysisResults 
            results={analysisResults} 
            isVisible={showAnalysis}
          />
        </div>

        <div className="mt-8 bg-blue-50 border border-blue-200 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-blue-900 mb-3">How to use:</h3>
          <ul className="space-y-2 text-blue-800">
            <li className="flex items-start">
              <span className="inline-block w-6 h-6 bg-blue-600 text-white text-sm rounded-full flex items-center justify-center mr-3 mt-0.5 flex-shrink-0">1</span>
              <span>Click "Start Camera" to activate your device camera</span>
            </li>
            <li className="flex items-start">
              <span className="inline-block w-6 h-6 bg-blue-600 text-white text-sm rounded-full flex items-center justify-center mr-3 mt-0.5 flex-shrink-0">2</span>
              <span>Position yourself in the camera preview</span>
            </li>
            <li className="flex items-start">
              <span className="inline-block w-6 h-6 bg-blue-600 text-white text-sm rounded-full flex items-center justify-center mr-3 mt-0.5 flex-shrink-0">3</span>
              <span>Click "Capture Image" to take a photo</span>
            </li>
            <li className="flex items-start">
              <span className="inline-block w-6 h-6 bg-blue-600 text-white text-sm rounded-full flex items-center justify-center mr-3 mt-0.5 flex-shrink-0">4</span>
              <span>Download your captured image or retake if needed</span>
            </li>
          </ul>
        </div>
      </div>
    </div>
  )
} 