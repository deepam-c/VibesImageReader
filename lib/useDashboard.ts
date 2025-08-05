'use client'

import { useState, useEffect, useCallback, useRef } from 'react'
import { DashboardData, getDashboardDataFromFirebase, setupFirebaseDashboardListener } from './firebase'

interface UseDashboardState {
  data: DashboardData | null
  loading: boolean
  error: string | null
  lastUpdated: Date | null
}

interface UseDashboardReturn extends UseDashboardState {
  refresh: () => void
  isConnected: boolean
  connectionType: 'firebase_realtime' | 'firebase_polling' | 'disconnected'
}

export function useDashboard(): UseDashboardReturn {
  const [state, setState] = useState<UseDashboardState>({
    data: null,
    loading: true,
    error: null,
    lastUpdated: null
  })
  
  const [isConnected, setIsConnected] = useState(false)
  const [connectionType, setConnectionType] = useState<'firebase_realtime' | 'firebase_polling' | 'disconnected'>('disconnected')
  const unsubscribeRef = useRef<(() => void) | null>(null)
  const isRefreshing = useRef(false)
  const isMountedRef = useRef(true)

  // Manual refresh function using direct Firebase
  const refresh = useCallback(async () => {
    if (isRefreshing.current || !isMountedRef.current) return
    
    try {
      isRefreshing.current = true
      setState(prev => ({ ...prev, loading: true, error: null }))
      
      console.log('🔄 Manually refreshing dashboard from Firebase...')
      const dashboardData = await getDashboardDataFromFirebase()
      
      if (isMountedRef.current) {
        setState({
          data: dashboardData,
          loading: false,
          error: null,
          lastUpdated: new Date()
        })
        setIsConnected(true)
        setConnectionType('firebase_polling')
      }
      
    } catch (error) {
      if (isMountedRef.current) {
        console.error('Error refreshing dashboard from Firebase:', error)
        setState(prev => ({
          ...prev,
          loading: false,
          error: error instanceof Error ? error.message : 'Failed to refresh dashboard'
        }))
        setIsConnected(false)
        setConnectionType('disconnected')
      }
    } finally {
      isRefreshing.current = false
    }
  }, [])

  // Setup Firebase real-time listener
  useEffect(() => {
    isMountedRef.current = true
    console.log('🚀 Setting up Firebase real-time dashboard connection...')
    
    // Setup real-time listener FIRST
    const handleRealtimeUpdate = (dashboardData: DashboardData) => {
      if (!isMountedRef.current) return
      
      console.log('🔥 Real-time Firebase update received!')
      setState({
        data: dashboardData,
        loading: false,
        error: null,
        lastUpdated: new Date()
      })
      setIsConnected(true)
      setConnectionType('firebase_realtime')
    }
    
    const handleError = (error: Error) => {
      if (!isMountedRef.current) return
      
      console.error('❌ Firebase real-time listener error:', error)
      setState(prev => ({
        ...prev,
        error: `Firebase connection error: ${error.message}`
      }))
      
      // Don't disconnect completely, keep last data but switch to polling mode
      setConnectionType('firebase_polling')
    }
    
    // Setup Firebase real-time listener immediately
    try {
      console.log('🔥 Setting up Firebase onSnapshot listener...')
      const unsubscribe = setupFirebaseDashboardListener(handleRealtimeUpdate, handleError)
      unsubscribeRef.current = unsubscribe
      
      console.log('✅ Firebase real-time dashboard listener active!')
      
      // The listener will automatically trigger with initial data
      // No need for separate initial load - onSnapshot fires immediately with current data
      
    } catch (error) {
      console.error('Failed to setup Firebase listener:', error)
      
      // Fallback: manual load if listener setup fails
      const loadFallbackData = async () => {
        try {
          console.log('📊 Fallback: Loading dashboard data from Firebase...')
          const dashboardData = await getDashboardDataFromFirebase()
          
          if (isMountedRef.current) {
            setState({
              data: dashboardData,
              loading: false,
              error: null,
              lastUpdated: new Date()
            })
            setIsConnected(true)
            setConnectionType('firebase_polling')
          }
        } catch (error) {
          console.error('Failed to load fallback dashboard data:', error)
          if (isMountedRef.current) {
            setState(prev => ({
              ...prev,
              loading: false,
              error: 'Failed to load dashboard data from Firebase'
            }))
            setIsConnected(false)
            setConnectionType('disconnected')
          }
        }
      }
      
      loadFallbackData()
    }

    // Cleanup function
    return () => {
      isMountedRef.current = false
      console.log('🛑 Cleaning up Firebase dashboard connection')
      
      if (unsubscribeRef.current) {
        unsubscribeRef.current()
        unsubscribeRef.current = null
      }
      
      setIsConnected(false)
      setConnectionType('disconnected')
    }
  }, []) // Only run once on mount

  return {
    ...state,
    refresh,
    isConnected,
    connectionType
  }
} 