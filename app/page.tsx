import Link from 'next/link'
import { 
  CameraIcon, 
  ChartBarIcon,
  EyeIcon,
  SparklesIcon,
  ClockIcon,
  UserGroupIcon
} from '@heroicons/react/24/outline'

export default function Home() {
  const stats = [
    { label: 'Images Analyzed', value: '2,847', change: '+12%' },
    { label: 'People Detected', value: '5,234', change: '+8%' },
    { label: 'Clothing Items', value: '12,567', change: '+23%' },
    { label: 'Accuracy Rate', value: '94.8%', change: '+2%' },
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

  const recentActivity = [
    { action: 'CV Analysis Completed', details: 'Formal outfit detected with accessories', time: '2 min ago' },
    { action: 'New Image Processed', details: 'Person detected with casual style', time: '5 min ago' },
    { action: 'Data Export Completed', details: 'Analytics report generated', time: '12 min ago' },
    { action: 'System Update', details: 'Enhanced clothing detection model', time: '1 hour ago' },
  ]

  return (
    <div className="fade-in">
      {/* Welcome Section */}
      <div className="corporate-card mb-8">
        <div className="corporate-card-content">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold text-gray-900 mb-2">
                Welcome to CV Analytics Pro
              </h1>
              <p className="text-gray-600 text-lg">
                Advanced computer vision analysis for clothing, accessories, and personal attributes
              </p>
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
              <span className="corporate-badge corporate-badge-success">All Systems Operational</span>
            </div>
            <div className="corporate-card-content">
              <div className="space-y-4">
                <div className="flex justify-between items-center">
                  <span className="text-gray-600">CV Backend</span>
                  <span className="corporate-badge corporate-badge-success">Online</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-600">Database</span>
                  <span className="corporate-badge corporate-badge-success">Connected</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-600">AI Models</span>
                  <span className="corporate-badge corporate-badge-success">Loaded</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-600">Processing Speed</span>
                  <span className="text-green-600 font-medium">~2.3s avg</span>
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
                    <span className="font-medium">94.8%</span>
                  </div>
                  <div className="corporate-progress">
                    <div className="corporate-progress-bar" style={{ width: '94.8%' }}></div>
                  </div>
                </div>
                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-gray-600">Processing Speed</span>
                    <span className="font-medium">87%</span>
                  </div>
                  <div className="corporate-progress">
                    <div className="corporate-progress-bar" style={{ width: '87%' }}></div>
                  </div>
                </div>
                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-gray-600">System Load</span>
                    <span className="font-medium">42%</span>
                  </div>
                  <div className="corporate-progress">
                    <div className="corporate-progress-bar" style={{ width: '42%' }}></div>
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