'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { 
  CameraIcon,
  ChartBarIcon,
  HomeIcon,
  UserCircleIcon
} from '@heroicons/react/24/outline'

const navigation = [
  { name: 'Dashboard', href: '/', icon: HomeIcon },
  { name: 'CV Analysis', href: '/capture', icon: CameraIcon },
  { name: 'Analytics', href: '/view-data', icon: ChartBarIcon },
]

export default function CorporateHeader() {
  const pathname = usePathname()

  return (
    <header className="corporate-header">
      <nav className="corporate-nav">
        <div className="corporate-logo">
          CV Analytics Pro
        </div>
        
        <div className="corporate-nav-items">
          {navigation.map((item) => {
            const isActive = pathname === item.href
            return (
              <Link
                key={item.name}
                href={item.href}
                className={`corporate-nav-item ${isActive ? 'active' : ''}`}
              >
                <item.icon className="w-5 h-5" />
                {item.name}
              </Link>
            )
          })}
        </div>

        <div className="flex items-center gap-4">
          <div className="corporate-badge corporate-badge-success">
            Beta
          </div>
          <UserCircleIcon className="w-8 h-8 text-white/80 hover:text-white cursor-pointer transition-colors" />
        </div>
      </nav>
    </header>
  )
} 