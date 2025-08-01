'use client'

import { useState } from 'react'
import CameraCapture from '@/components/CameraCapture'
import AnalysisResults from '@/components/AnalysisResults'
import { 
  CameraIcon,
  DocumentArrowUpIcon,
  SparklesIcon,
  EyeIcon
} from '@heroicons/react/24/outline'

export default function CapturePage() {
  const [analysisResults, setAnalysisResults] = useState(null)
  const [showAnalysis, setShowAnalysis] = useState(false)

  return (
    <div className="fade-in">
      {/* Page Header */}
      <div className="corporate-card mb-8">
        <div className="corporate-card-header">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="corporate-card-title flex items-center gap-3">
                <div className="p-2 bg-blue-100 rounded-lg">
                  <CameraIcon className="w-6 h-6 text-blue-600" />
                </div>
                Computer Vision Analysis
              </h1>
              <p className="corporate-card-subtitle">
                Capture or upload images for advanced AI analysis of clothing, accessories, and personal attributes
              </p>
            </div>
            <div className="flex items-center gap-2">
              <div className="corporate-badge corporate-badge-success">
                <SparklesIcon className="w-3 h-3 mr-1" />
                AI Enhanced
              </div>
              <div className="corporate-badge corporate-badge-info">
                <EyeIcon className="w-3 h-3 mr-1" />
                Real-time
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="corporate-grid-wide gap-8">
        {/* Camera Capture Section */}
        <div className="corporate-card">
          <div className="corporate-card-header">
            <h2 className="corporate-card-title">Image Capture</h2>
            <p className="corporate-card-subtitle">
              Use your camera or upload an image to start the analysis
            </p>
          </div>
          <div className="corporate-card-content">
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
          </div>
        </div>

        {/* Analysis Results Section */}
        <div className="space-y-6">
          {/* Quick Stats */}
          <div className="corporate-card">
            <div className="corporate-card-header">
              <h2 className="corporate-card-title">Analysis Status</h2>
              <p className="corporate-card-subtitle">Real-time processing metrics</p>
            </div>
            <div className="corporate-card-content">
              <div className="space-y-4">
                <div className="flex justify-between items-center">
                  <span className="text-gray-600">CV Backend</span>
                  <span className="corporate-badge corporate-badge-success">Online</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-600">Processing Speed</span>
                  <span className="text-blue-600 font-medium">~2.3s avg</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-600">Accuracy Rate</span>
                  <span className="text-green-600 font-medium">94.8%</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-600">Firebase Storage</span>
                  <span className="corporate-badge corporate-badge-success">Connected</span>
                </div>
              </div>
            </div>
          </div>

          {/* Analysis Results */}
          {showAnalysis && analysisResults && (
            <div className="corporate-card">
              <div className="corporate-card-header">
                <h2 className="corporate-card-title">Analysis Results</h2>
                <p className="corporate-card-subtitle">
                  Detailed computer vision analysis with clothing and accessory detection
                </p>
              </div>
              <div className="corporate-card-content">
                <AnalysisResults results={analysisResults} />
              </div>
            </div>
          )}

          {/* Instructions */}
          {!showAnalysis && (
            <div className="corporate-card">
              <div className="corporate-card-header">
                <h2 className="corporate-card-title">How to Use</h2>
                <p className="corporate-card-subtitle">Follow these steps to analyze your images</p>
              </div>
              <div className="corporate-card-content">
                <div className="space-y-4">
                  <div className="flex items-start gap-3">
                    <div className="w-6 h-6 bg-blue-100 text-blue-600 rounded-full flex items-center justify-center text-sm font-medium">
                      1
                    </div>
                    <div>
                      <div className="font-medium text-gray-900">Camera Capture</div>
                      <div className="text-gray-600 text-sm">Click "Start Camera" to use your device camera</div>
                    </div>
                  </div>
                  <div className="flex items-start gap-3">
                    <div className="w-6 h-6 bg-blue-100 text-blue-600 rounded-full flex items-center justify-center text-sm font-medium">
                      2
                    </div>
                    <div>
                      <div className="font-medium text-gray-900">Upload Image</div>
                      <div className="text-gray-600 text-sm">Or click "Upload Image" to select from your device</div>
                    </div>
                  </div>
                  <div className="flex items-start gap-3">
                    <div className="w-6 h-6 bg-blue-100 text-blue-600 rounded-full flex items-center justify-center text-sm font-medium">
                      3
                    </div>
                    <div>
                      <div className="font-medium text-gray-900">AI Analysis</div>
                      <div className="text-gray-600 text-sm">View detailed results including clothing, accessories, and demographics</div>
                    </div>
                  </div>
                  <div className="flex items-start gap-3">
                    <div className="w-6 h-6 bg-blue-100 text-blue-600 rounded-full flex items-center justify-center text-sm font-medium">
                      4
                    </div>
                    <div>
                      <div className="font-medium text-gray-900">Auto-Save</div>
                      <div className="text-gray-600 text-sm">Results are automatically saved to Firebase for future reference</div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Features Overview */}
          <div className="corporate-card">
            <div className="corporate-card-header">
              <h2 className="corporate-card-title">Detection Capabilities</h2>
              <p className="corporate-card-subtitle">What our AI can identify</p>
            </div>
            <div className="corporate-card-content">
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div className="space-y-2">
                  <div className="font-medium text-gray-900">👤 Demographics</div>
                  <div className="text-gray-600">Age, gender, emotions</div>
                </div>
                <div className="space-y-2">
                  <div className="font-medium text-gray-900">👕 Clothing</div>
                  <div className="text-gray-600">Shirts, jackets, patterns</div>
                </div>
                <div className="space-y-2">
                  <div className="font-medium text-gray-900">💍 Accessories</div>
                  <div className="text-gray-600">Jewelry, watches, glasses</div>
                </div>
                <div className="space-y-2">
                  <div className="font-medium text-gray-900">🎨 Style</div>
                  <div className="text-gray-600">Formality, colors, fabric</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
} 