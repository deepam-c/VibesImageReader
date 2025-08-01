'use client'

import { useState, useEffect } from 'react'
import { getCVAnalyses, CVAnalysisData } from '@/lib/firebase'
import { 
  ChartBarIcon,
  MagnifyingGlassIcon,
  FunnelIcon,
  DocumentDuplicateIcon,
  ClockIcon,
  EyeIcon,
  UserIcon,
  SparklesIcon
} from '@heroicons/react/24/outline'

export default function ViewDataPage() {
  const [analyses, setAnalyses] = useState<CVAnalysisData[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [selectedAnalysis, setSelectedAnalysis] = useState<CVAnalysisData | null>(null)
  const [searchTerm, setSearchTerm] = useState('')
  const [filterBy, setFilterBy] = useState<'all' | 'formal' | 'casual'>('all')

  useEffect(() => {
    loadAnalyses()
  }, [])

  const loadAnalyses = async () => {
    try {
      setLoading(true)
      setError(null)
      const data = await getCVAnalyses()
      setAnalyses(data)
    } catch (err) {
      console.error('Failed to load analyses:', err)
      setError('Failed to load saved analyses. Please check your Firebase configuration.')
    } finally {
      setLoading(false)
    }
  }

  const filteredAnalyses = analyses.filter(analysis => {
    const matchesSearch = searchTerm === '' || 
      JSON.stringify(analysis).toLowerCase().includes(searchTerm.toLowerCase())

    const matchesFilter = filterBy === 'all' || 
      (analysis.peopleAnalysis && analysis.peopleAnalysis.some(person => 
        person.appearance?.outfit_formality?.toLowerCase().includes(filterBy)
      ))

    return matchesSearch && matchesFilter
  })

  const formatTimestamp = (timestamp: any) => {
    if (timestamp?.toDate) {
      return timestamp.toDate().toLocaleString()
    }
    return new Date(timestamp).toLocaleString()
  }

  const formatJSON = (data: any) => {
    // Convert any objects that might have non-serializable values
    const cleanData = JSON.parse(JSON.stringify(data, (key, value) => {
      // Convert any objects with prediction/confidence to strings
      if (typeof value === 'object' && value !== null && 'prediction' in value && 'confidence' in value) {
        return `${value.prediction} (${(value.confidence * 100).toFixed(1)}%)`
      }
      // Convert Firestore timestamps to readable strings
      if (value && typeof value === 'object' && value.toDate) {
        return value.toDate().toISOString()
      }
      return value
    }))
    return JSON.stringify(cleanData, null, 2)
  }

  if (loading) {
    return (
      <div className="fade-in">
        <div className="corporate-card">
          <div className="corporate-card-content">
            <div className="flex items-center justify-center py-12">
              <div className="corporate-loading">
                <div className="corporate-spinner"></div>
                <span className="text-lg">Loading saved analyses...</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="fade-in">
      {/* Page Header */}
      <div className="corporate-card mb-8">
        <div className="corporate-card-header">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="corporate-card-title flex items-center gap-3">
                <div className="p-2 bg-purple-100 rounded-lg">
                  <ChartBarIcon className="w-6 h-6 text-purple-600" />
                </div>
                Analytics Dashboard
              </h1>
              <p className="corporate-card-subtitle">
                View and explore your saved computer vision analysis data
              </p>
            </div>
            <div className="flex items-center gap-2">
              <div className="corporate-badge corporate-badge-info">
                {filteredAnalyses.length} analyses
              </div>
              <button 
                onClick={loadAnalyses}
                className="btn-corporate-secondary"
              >
                Refresh Data
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Error State */}
      {error && (
        <div className="corporate-card mb-6 border-red-200 bg-red-50">
          <div className="corporate-card-content">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-red-100 rounded-lg flex items-center justify-center">
                <span className="text-red-600">⚠️</span>
              </div>
              <div className="flex-1">
                <div className="font-medium text-red-900">Connection Error</div>
                <div className="text-red-700 text-sm">{error}</div>
              </div>
              <button 
                onClick={loadAnalyses}
                className="btn-corporate-primary"
              >
                Retry
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Stats Overview */}
      <div className="corporate-stats-grid mb-8">
        <div className="corporate-stat-card">
          <div className="corporate-stat-value">{analyses.length}</div>
          <div className="corporate-stat-label">Total Analyses</div>
        </div>
                 <div className="corporate-stat-card">
           <div className="corporate-stat-value">
             {analyses.reduce((sum, analysis) => sum + (analysis.detectionSummary?.peopleDetected || 0), 0)}
           </div>
           <div className="corporate-stat-label">People Detected</div>
         </div>
         <div className="corporate-stat-card">
           <div className="corporate-stat-value">
             {analyses.length > 0 ? (analyses.reduce((sum, analysis) => sum + (analysis.detectionSummary?.averageConfidence || 0), 0) / analyses.length * 100).toFixed(1) + '%' : '0%'}
           </div>
           <div className="corporate-stat-label">Avg Confidence</div>
         </div>
         <div className="corporate-stat-card">
           <div className="corporate-stat-value">
             {analyses.length > 0 ? Math.round(analyses.reduce((sum, analysis) => sum + (analysis.processingInfo?.processingTime || 0), 0) / analyses.length) : 0}ms
           </div>
           <div className="corporate-stat-label">Avg Processing Time</div>
         </div>
      </div>

      {/* Search and Filter Controls */}
      <div className="corporate-card mb-6">
        <div className="corporate-card-content">
          <div className="flex flex-col sm:flex-row gap-4">
            <div className="flex-1 relative">
              <MagnifyingGlassIcon className="w-5 h-5 absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
              <input
                type="text"
                placeholder="Search analyses..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="corporate-input pl-10"
              />
            </div>
            <div className="flex items-center gap-2">
              <FunnelIcon className="w-5 h-5 text-gray-400" />
              <select
                value={filterBy}
                onChange={(e) => setFilterBy(e.target.value as 'all' | 'formal' | 'casual')}
                className="corporate-select"
              >
                <option value="all">All Styles</option>
                <option value="formal">Formal</option>
                <option value="casual">Casual</option>
              </select>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      {filteredAnalyses.length === 0 ? (
        <div className="corporate-card">
          <div className="corporate-card-content">
            <div className="text-center py-12">
              <div className="w-16 h-16 bg-gray-100 rounded-lg flex items-center justify-center mx-auto mb-4">
                <ChartBarIcon className="w-8 h-8 text-gray-400" />
              </div>
              <h3 className="text-lg font-medium text-gray-900 mb-2">No analyses found</h3>
              <p className="text-gray-600 mb-6">
                {analyses.length === 0 
                  ? "Start by capturing some images to see your analysis data here."
                  : "Try adjusting your search or filter criteria."
                }
              </p>
              <a href="/capture" className="btn-corporate-primary">
                Start Analysis
              </a>
            </div>
          </div>
        </div>
      ) : (
        <div className="corporate-grid-wide gap-8">
          {/* Analysis List */}
          <div className="space-y-4">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-semibold text-gray-900">Analysis History</h2>
              <span className="corporate-badge corporate-badge-info">
                {filteredAnalyses.length} results
              </span>
            </div>
            
            {filteredAnalyses.map((analysis) => (
              <div
                key={analysis.id}
                onClick={() => setSelectedAnalysis(analysis)}
                className={`corporate-card cursor-pointer transition-all ${
                  selectedAnalysis?.id === analysis.id
                    ? 'ring-2 ring-blue-500 bg-blue-50'
                    : 'hover:shadow-lg'
                }`}
              >
                <div className="corporate-card-content">
                  <div className="flex justify-between items-start mb-3">
                    <div className="flex items-center gap-3">
                      <div className="w-10 h-10 bg-blue-100 rounded-lg flex items-center justify-center">
                        <EyeIcon className="w-5 h-5 text-blue-600" />
                      </div>
                      <div>
                        <h3 className="font-medium text-gray-900">
                          Analysis #{analysis.id?.slice(-6)}
                        </h3>
                        <div className="flex items-center gap-2 text-sm text-gray-500">
                          <ClockIcon className="w-4 h-4" />
                          {formatTimestamp(analysis.timestamp)}
                        </div>
                      </div>
                    </div>
                    <div className="flex gap-2">
                      <span className="corporate-badge corporate-badge-success">
                        {(analysis.detectionSummary.averageConfidence * 100).toFixed(0)}%
                      </span>
                    </div>
                  </div>
                  
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 text-sm mb-4">
                    <div className="flex items-center gap-2">
                      <UserIcon className="w-4 h-4 text-gray-400" />
                      <span className="text-gray-600">People:</span>
                      <span className="font-medium">{analysis.detectionSummary.peopleDetected}</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <EyeIcon className="w-4 h-4 text-gray-400" />
                      <span className="text-gray-600">Faces:</span>
                      <span className="font-medium">{analysis.detectionSummary.facesAnalyzed}</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <SparklesIcon className="w-4 h-4 text-gray-400" />
                      <span className="text-gray-600">Confidence:</span>
                      <span className="font-medium">{(analysis.detectionSummary.averageConfidence * 100).toFixed(1)}%</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <ClockIcon className="w-4 h-4 text-gray-400" />
                      <span className="text-gray-600">Time:</span>
                      <span className="font-medium">{analysis.processingInfo.processingTime}ms</span>
                    </div>
                  </div>

                                      {analysis.peopleAnalysis && analysis.peopleAnalysis.length > 0 && (
                      <div className="flex flex-wrap gap-2">
                        {analysis.peopleAnalysis[0]?.appearance?.clothing?.detected_items?.slice(0, 3).map((item, idx) => (
                          <span key={idx} className="corporate-badge corporate-badge-info">
                            {typeof item === 'string' ? item.replace('_', ' ') : 'Item'}
                          </span>
                        ))}
                        {analysis.peopleAnalysis[0]?.appearance?.accessories?.slice(0, 2).map((accessory, idx) => (
                          <span key={idx} className="corporate-badge corporate-badge-purple">
                            {typeof accessory === 'string' ? accessory.replace('_', ' ') : 'Accessory'}
                          </span>
                        ))}
                      </div>
                    )}
                </div>
              </div>
            ))}
          </div>

          {/* Detailed View */}
          <div className="lg:sticky lg:top-4">
            {selectedAnalysis ? (
              <div className="corporate-card">
                <div className="corporate-card-header">
                  <div className="flex items-center justify-between">
                    <h2 className="corporate-card-title">Analysis Details</h2>
                    <button
                      onClick={() => {
                        navigator.clipboard.writeText(formatJSON(selectedAnalysis))
                        // You could add a toast notification here
                        alert('JSON copied to clipboard!')
                      }}
                      className="btn-corporate-secondary"
                    >
                      <DocumentDuplicateIcon className="w-4 h-4" />
                      Copy JSON
                    </button>
                  </div>
                </div>
                
                <div className="corporate-card-content space-y-6">
                  {/* Metadata */}
                  <div>
                    <h3 className="font-medium text-gray-900 mb-3">Metadata</h3>
                    <div className="bg-gray-50 rounded-lg p-4 space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span className="text-gray-600">Timestamp:</span>
                        <span className="font-medium">{formatTimestamp(selectedAnalysis.timestamp)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600">Processing Time:</span>
                        <span className="font-medium">{selectedAnalysis.processingInfo?.processingTime || 0}ms</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600">Model Version:</span>
                        <span className="font-medium">{selectedAnalysis.processingInfo?.modelVersion || 'Unknown'}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600">Image Size:</span>
                        <span className="font-medium">{((selectedAnalysis.imageMetadata?.size || 0) / 1024).toFixed(1)}KB</span>
                      </div>
                    </div>
                  </div>

                  {/* People Analysis Summary */}
                                        {selectedAnalysis.peopleAnalysis && selectedAnalysis.peopleAnalysis.length > 0 && (
                        <div>
                          <h3 className="font-medium text-gray-900 mb-3">People Analysis</h3>
                          {selectedAnalysis.peopleAnalysis.map((person, idx) => (
                            <div key={idx} className="bg-gray-50 rounded-lg p-4 mb-3">
                              <div className="space-y-2 text-sm">
                                <div className="flex justify-between">
                                  <span className="text-gray-600">Age:</span>
                                  <span className="font-medium">
                                    {person.demographics?.estimatedAge || 'Unknown'} 
                                    {person.demographics?.ageRange ? ` (${person.demographics.ageRange})` : ''}
                                  </span>
                                </div>
                                <div className="flex justify-between">
                                  <span className="text-gray-600">Gender:</span>
                                  <span className="font-medium capitalize">{person.demographics?.gender || 'Unknown'}</span>
                                </div>
                                <div className="flex justify-between">
                                  <span className="text-gray-600">Emotion:</span>
                                  <span className="font-medium capitalize">{person.emotions?.primary || 'Unknown'}</span>
                                </div>
                                <div className="flex justify-between">
                                  <span className="text-gray-600">Style:</span>
                                  <span className="font-medium capitalize">
                                    {person.appearance?.outfit_formality ? person.appearance.outfit_formality.replace('_', ' ') : 'Unknown'}
                                  </span>
                                </div>
                                {/* Clothing Items */}
                                {person.appearance?.clothing?.detected_items && person.appearance.clothing.detected_items.length > 0 && (
                                  <div className="flex justify-between">
                                    <span className="text-gray-600">Clothing:</span>
                                    <div className="flex flex-wrap gap-1">
                                      {person.appearance.clothing.detected_items.slice(0, 3).map((item, itemIdx) => (
                                        <span key={itemIdx} className="px-2 py-1 bg-blue-100 text-blue-800 text-xs rounded">
                                          {typeof item === 'string' ? item.replace('_', ' ') : 'Item'}
                                        </span>
                                      ))}
                                    </div>
                                  </div>
                                )}
                                {/* Accessories */}
                                {person.appearance?.accessories && person.appearance.accessories.length > 0 && (
                                  <div className="flex justify-between">
                                    <span className="text-gray-600">Accessories:</span>
                                    <div className="flex flex-wrap gap-1">
                                      {person.appearance.accessories.slice(0, 3).map((accessory, accIdx) => (
                                        <span key={accIdx} className="px-2 py-1 bg-purple-100 text-purple-800 text-xs rounded">
                                          {typeof accessory === 'string' ? accessory.replace('_', ' ') : 'Accessory'}
                                        </span>
                                      ))}
                                    </div>
                                  </div>
                                )}
                              </div>
                            </div>
                          ))}
                        </div>
                      )}

                  {/* Raw JSON */}
                  <div>
                    <h3 className="font-medium text-gray-900 mb-3">Raw JSON Data</h3>
                    <div className="bg-gray-900 text-gray-100 p-4 rounded-lg">
                      <pre className="text-xs overflow-auto max-h-96 whitespace-pre-wrap">
                        {formatJSON(selectedAnalysis)}
                      </pre>
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              <div className="corporate-card">
                <div className="corporate-card-content">
                  <div className="text-center py-12">
                    <div className="w-16 h-16 bg-gray-100 rounded-lg flex items-center justify-center mx-auto mb-4">
                      <EyeIcon className="w-8 h-8 text-gray-400" />
                    </div>
                    <h3 className="font-medium text-gray-900 mb-2">Select an Analysis</h3>
                    <p className="text-gray-600">Click on any analysis from the list to view detailed information</p>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
} 