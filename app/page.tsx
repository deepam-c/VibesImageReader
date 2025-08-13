'use client'

import Link from 'next/link'
import { 
  CameraIcon, 
  ChartBarIcon,
  EyeIcon,
  SparklesIcon,
  ClockIcon,
  UserGroupIcon,
  ArrowPathIcon,
  WifiIcon,
  ExclamationTriangleIcon
} from '@heroicons/react/24/outline'
import { useDashboard } from '@/lib/useDashboard'

export default function Home() {
  const { data: dashboardData, loading, error, refresh, isConnected, connectionType, lastUpdated } = useDashboard()
  
  // Add connection status indicator
  const getConnectionStatusColor = () => {
    if (!isConnected) return 'bg-red-500'
    if (connectionType === 'firebase_realtime') return 'bg-green-500'
    if (connectionType === 'firebase_polling') return 'bg-blue-500'
    return 'bg-gray-500'
  }
  
  const getConnectionStatusText = () => {
    if (!isConnected) return 'Disconnected'
    if (connectionType === 'firebase_realtime') return 'Firebase Real-time'
    if (connectionType === 'firebase_polling') return 'Firebase Connected'
    return 'Connected'
  }
  
  const getConnectionDescription = () => {
    if (!isConnected) return 'No connection to Firebase'
    if (connectionType === 'firebase_realtime') return 'Real-time Updates • Firebase onSnapshot'
    if (connectionType === 'firebase_polling') return 'Firebase Direct • Live Data'
    return 'Connected'
  }
  
  // Use real-time data or show loading/error state
  const data = dashboardData
  
  // Show loading state if no data and still loading
  if (loading && !data) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-4 text-lg text-gray-600">Connecting to dashboard...</p>
          <p className="text-sm text-gray-500">Status: {getConnectionStatusText()}</p>
        </div>
      </div>
    )
  }
  
  // Show error state if there's an error and no data
  if (error && !data) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center max-w-md">
          <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded">
            <h3 className="font-bold">Connection Error</h3>
            <p className="text-sm">{error}</p>
            <p className="text-xs mt-2">Status: {getConnectionStatusText()}</p>
          </div>
          <button
            onClick={refresh}
            className="mt-4 bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700"
          >
            Retry Connection
          </button>
        </div>
      </div>
    )
  }
  
  // Show placeholder state if no data available
  if (!data) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <p className="text-lg text-gray-600">No dashboard data available</p>
          <p className="text-sm text-gray-500">Status: {getConnectionStatusText()}</p>
          <button
            onClick={refresh}
            className="mt-4 bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700"
          >
            Load Dashboard
          </button>
        </div>
      </div>
    )
  }
  
  const stats = [
    {
      label: 'Images Analyzed',
      value: data.stats?.images_analyzed?.value || '0',
      change: data.stats?.images_analyzed?.change || '+0%'
    },
    {
      label: 'People Detected',
      value: data.stats?.people_detected?.value || '0',
      change: data.stats?.people_detected?.change || '+0%'
    },
    {
      label: 'Clothing Items',
      value: data.stats?.clothing_items?.value || '0',
      change: data.stats?.clothing_items?.change || '+0%'
    },
    {
      label: 'Avg Confidence',
      value: data.stats?.accuracy_rate?.value || '0%',
      change: data.stats?.accuracy_rate?.change || '+0%'
    }
  ]

  const features = [
    {
      title: 'Advanced CV Analysis',
      description: 'Real-time computer vision analysis with clothing and accessory detection',
      icon: EyeIcon,
      href: '/capture',
      color: 'corporate-badge-info'
    },
    {
      title: 'Analytics Dashboard',
      description: 'Comprehensive analytics and historical data visualization',
      icon: ChartBarIcon,
      href: '/view-data',
      color: 'corporate-badge-purple'
    },
    {
      title: 'Smart Detection',
      description: 'AI-powered detection of demographics, emotions, and style attributes',
      icon: SparklesIcon,
      href: '/capture',
      color: 'corporate-badge-success'
    }
  ]

  const recentActivity = data.recent_activity

  return (
    <div className="fade-in">
      {/* Welcome Section */}
      <div className="corporate-card mb-8">
        <div className="corporate-card-content">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold text-gray-900 mb-2">
                Welcome to AI Eye
              </h1>
              <p className="text-gray-600 text-lg">
                Advanced computer vision analysis for clothing, accessories, and personal attributes
              </p>
              
              {/* Real-time status indicator */}
              <div className="flex items-center gap-4 mt-3">
                <div className="flex items-center gap-2">
                  <div className={`w-2 h-2 rounded-full ${getConnectionStatusColor()} ${isConnected ? 'animate-pulse' : ''}`}></div>
                  <WifiIcon className={`w-4 h-4 ${isConnected ? (connectionType === 'firebase_realtime' ? 'text-green-500' : 'text-blue-500') : 'text-red-500'}`} />
                  <span className={`text-sm font-medium ${isConnected ? (connectionType === 'firebase_realtime' ? 'text-green-600' : 'text-blue-600') : 'text-red-600'}`}>
                    {isConnected ? (connectionType === 'firebase_realtime' ? '🟢 LIVE' : '🟡 LIVE') : '🔴 OFFLINE'}
                  </span>
                  <span className="text-gray-400 text-xs">•</span>
                  <span className="text-sm text-gray-600">
                    {getConnectionDescription()}
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  <ClockIcon className="w-4 h-4 text-gray-400" />
                  <span className="text-sm text-gray-500">
                    {lastUpdated ? lastUpdated.toLocaleTimeString() : 'Never'}
                  </span>
                </div>
                <button
                  onClick={refresh}
                  disabled={loading}
                  className="flex items-center gap-1 text-blue-600 hover:text-blue-700 text-sm font-medium disabled:opacity-50"
                >
                  <ArrowPathIcon className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
                  Manual Refresh
                </button>
              </div>
            </div>
            <div className="flex gap-3">
              <Link href="/capture" className="btn-corporate-primary">
                <CameraIcon className="w-5 h-5" />
                Start Analysis
              </Link>
              <Link href="/view-data" className="btn-corporate-secondary">
                <ChartBarIcon className="w-5 h-5" />
                View Analytics
              </Link>
            </div>
          </div>
        </div>
      </div>

      {/* Stats Grid */}
      <div className="corporate-stats-grid mb-8">
        {stats.map((stat, index) => (
          <div key={index} className="corporate-stat-card slide-up" style={{ animationDelay: `${index * 0.1}s` }}>
            <div className="corporate-stat-value">{stat.value}</div>
            <div className="corporate-stat-label">{stat.label}</div>
            <div className="text-green-600 text-sm font-medium mt-1">{stat.change}</div>
          </div>
        ))}
      </div>

      {/* Main Content Grid */}
      <div className="corporate-grid-wide gap-8">
        {/* Features Section */}
        <div className="space-y-6">
          <div className="corporate-card">
            <div className="corporate-card-header">
              <h2 className="corporate-card-title">Key Features</h2>
              <p className="corporate-card-subtitle">Explore our powerful computer vision capabilities</p>
            </div>
            <div className="corporate-card-content">
              <div className="space-y-4">
                {features.map((feature, index) => (
                  <Link 
                    key={index} 
                    href={feature.href}
                    className="block p-4 border border-gray-200 rounded-lg hover:border-blue-300 transition-all hover:bg-blue-50"
                  >
                    <div className="flex items-start gap-4">
                      <div className="p-2 bg-blue-100 rounded-lg">
                        <feature.icon className="w-6 h-6 text-blue-600" />
                      </div>
                      <div className="flex-1">
                        <div className="flex items-center gap-2 mb-1">
                          <h3 className="font-semibold text-gray-900">{feature.title}</h3>
                          <span className={`corporate-badge ${feature.color}`}>New</span>
                        </div>
                        <p className="text-gray-600 text-sm">{feature.description}</p>
                      </div>
                    </div>
                  </Link>
                ))}
              </div>
            </div>
          </div>

          {/* Quick Actions */}
          <div className="corporate-card">
            <div className="corporate-card-header">
              <h2 className="corporate-card-title">Quick Actions</h2>
              <p className="corporate-card-subtitle">Get started with these common tasks</p>
            </div>
            <div className="corporate-card-content">
              <div className="grid grid-cols-2 gap-3">
                <Link href="/capture" className="p-4 border border-gray-200 rounded-lg text-center hover:border-blue-300 transition-all hover:bg-blue-50">
                  <CameraIcon className="w-8 h-8 text-blue-600 mx-auto mb-2" />
                  <div className="font-medium text-gray-900">Capture Image</div>
                  <div className="text-xs text-gray-500">Take or upload photo</div>
                </Link>
                <Link href="/view-data" className="p-4 border border-gray-200 rounded-lg text-center hover:border-blue-300 transition-all hover:bg-blue-50">
                  <ChartBarIcon className="w-8 h-8 text-purple-600 mx-auto mb-2" />
                  <div className="font-medium text-gray-900">View Data</div>
                  <div className="text-xs text-gray-500">Browse analytics</div>
                </Link>
              </div>
            </div>
          </div>
        </div>

        {/* Side Panel */}
        <div className="space-y-6">
          {/* System Status */}
          <div className="corporate-card">
            <div className="corporate-card-header">
              <h2 className="corporate-card-title">System Status</h2>
              <span className={`corporate-badge ${
                data.system_status.overall === 'All Systems Operational' 
                  ? 'corporate-badge-success' 
                  : 'corporate-badge-warning'
              }`}>
                {data.system_status.overall}
              </span>
            </div>
            <div className="corporate-card-content">
              <div className="space-y-4">
                <div className="flex justify-between items-center">
                  <span className="text-gray-600">CV Backend</span>
                  <span className={`corporate-badge ${
                    data.system_status.cv_backend === 'Online' 
                      ? 'corporate-badge-success' 
                      : 'corporate-badge-error'
                  }`}>
                    {data.system_status.cv_backend}
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-600">Database</span>
                  <span className={`corporate-badge ${
                    data.system_status.database === 'Connected' 
                      ? 'corporate-badge-success' 
                      : 'corporate-badge-error'
                  }`}>
                    {data.system_status.database}
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-600">AI Models</span>
                  <span className={`corporate-badge ${
                    data.system_status.ai_models === 'Loaded' 
                      ? 'corporate-badge-success' 
                      : 'corporate-badge-error'
                  }`}>
                    {data.system_status.ai_models}
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-600">Processing Speed</span>
                  <span className="text-green-600 font-medium">{data.system_status.processing_speed}</span>
                </div>
              </div>
            </div>
          </div>

          {/* Recent Activity */}
          <div className="corporate-card">
            <div className="corporate-card-header">
              <h2 className="corporate-card-title">Recent Activity</h2>
              <p className="corporate-card-subtitle">Latest system events</p>
            </div>
            <div className="corporate-card-content">
              <div className="space-y-3">
                {recentActivity.map((activity, index) => (
                  <div key={index} className="flex items-start gap-3 p-3 bg-gray-50 rounded-lg">
                    <ClockIcon className="w-4 h-4 text-gray-400 mt-0.5" />
                    <div className="flex-1">
                      <div className="font-medium text-gray-900 text-sm">{activity.action}</div>
                      <div className="text-gray-600 text-xs">{activity.details}</div>
                      <div className="text-gray-400 text-xs mt-1">{activity.time}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Performance Metrics */}
          <div className="corporate-card">
            <div className="corporate-card-header">
              <h2 className="corporate-card-title">Performance</h2>
              <p className="corporate-card-subtitle">Real-time metrics</p>
            </div>
            <div className="corporate-card-content">
              <div className="space-y-4">
                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-gray-600">Detection Accuracy</span>
                    <span className="font-medium">{data.performance_metrics.detection_accuracy.value}</span>
                  </div>
                  <div className="corporate-progress">
                    <div 
                      className="corporate-progress-bar" 
                      style={{ width: `${data.performance_metrics.detection_accuracy.percentage}%` }}
                    ></div>
                  </div>
                </div>
                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-gray-600">Processing Speed</span>
                    <span className="font-medium">{data.performance_metrics.processing_speed.value}</span>
                  </div>
                  <div className="corporate-progress">
                    <div 
                      className="corporate-progress-bar" 
                      style={{ width: `${data.performance_metrics.processing_speed.percentage}%` }}
                    ></div>
                  </div>
                </div>
                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-gray-600">System Load</span>
                    <span className="font-medium">{data.performance_metrics.system_load.value}</span>
                  </div>
                  <div className="corporate-progress">
                    <div 
                      className="corporate-progress-bar" 
                      style={{ width: `${data.performance_metrics.system_load.percentage}%` }}
                    ></div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
} 