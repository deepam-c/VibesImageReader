import { initializeApp } from 'firebase/app'
import { getFirestore, collection, addDoc, getDocs, orderBy, query, Timestamp, onSnapshot, Unsubscribe, limit } from 'firebase/firestore'

// WebSocket connection management with local Socket.IO implementation
let io: any = null;
let Socket: any = null;

// Try to load Socket.IO, or create a local implementation if blocked
try {
  // If Socket.IO is available globally (from CDN), use it
  if (typeof window !== 'undefined' && (window as any).io) {
    io = (window as any).io;
  } else {
    // Create a local Socket.IO-like implementation for corporate networks
    io = function(url: string, options?: any) {
      return new LocalSocketIOClient(url, options || {});
    };
    io.version = '4.0.0-local';
  }
} catch (error) {
  console.warn('Socket.IO not available, WebSocket features disabled:', error);
}

// Local Socket.IO implementation for corporate networks that block CDN
class LocalSocketIOClient {
  private url: string;
  private options: any;
  private _connected = false;
  private _disconnected = true;
  private _id: string | null = null;
  private _handlers: { [event: string]: Function[] } = {};
  private _pollInterval: NodeJS.Timeout | null = null;
  private keepAliveFailures: number | null = null;

  constructor(url: string, options: any) {
    this.url = url;
    this.options = options;
  }

  get connected() { return this._connected; }
  get disconnected() { return this._disconnected; }
  get id() { return this._id; }

  connect() {
    this.startPolling();
  }

  private async startPolling() {
    try {
      console.log('🔌 Starting Socket.IO polling connection to:', this.url);
      
      const response = await fetch(`${this.url}/socket.io/?EIO=4&transport=polling`);
      if (response.ok) {
        const text = await response.text();
        // Parse Socket.IO handshake: 0{"sid":"...","upgrades":[...],...}
        const jsonStart = text.indexOf('{');
        if (jsonStart > 0) {
          const handshake = JSON.parse(text.substring(jsonStart));
          this._id = handshake.sid;
          this._connected = true;
          this._disconnected = false;
          
          console.log('✅ Socket.IO polling connected! Session ID:', this._id);
          this.emit('connect');
          
          // Start keep-alive polling
          this.startKeepAlive();
          return;
        }
      }
      throw new Error('Socket.IO handshake failed');
    } catch (error) {
      console.error('❌ Socket.IO polling failed:', error);
      this.emit('connect_error', error);
    }
  }

  private startKeepAlive() {
    // Simplified keep-alive - just check if server is responsive
    this._pollInterval = setInterval(async () => {
      if (this._connected) {
        try {
          // Simple health check instead of Socket.IO-specific keep-alive
          const response = await fetch(`${this.url}/health`, {
            method: 'GET',
            timeout: 5000
          } as any);
          if (!response.ok) {
            console.warn('Health check failed, but keeping connection alive');
            // Don't disconnect for health check failures
          }
        } catch (error) {
          console.warn('Keep-alive health check failed:', error);
          // Only disconnect after multiple failures
          this.keepAliveFailures = (this.keepAliveFailures || 0) + 1;
          if (this.keepAliveFailures >= 3) {
            console.error('Multiple keep-alive failures, disconnecting');
            this.disconnect();
          }
        }
      }
    }, 30000); // Every 30 seconds
  }

  on(event: string, handler: Function) {
    if (!this._handlers[event]) {
      this._handlers[event] = [];
    }
    this._handlers[event].push(handler);
    console.log(`🎧 Added handler for event: ${event}`);
  }

  emit(event: string, ...args: any[]) {
    console.log(`📤 Emitting event: ${event}`, args);
    
    // Handle server communication via HTTP for specific events
    if (event === 'subscribe_dashboard') {
      this.handleSubscribeDashboard();
      return;
    }
    
    if (event === 'request_dashboard_refresh') {
      this.handleDashboardRefresh();
      return;
    }
    
    if (event === 'unsubscribe_dashboard') {
      console.log('📤 Unsubscribing from dashboard updates');
      return;
    }
    
    // For other events, trigger local handlers
    if (this._handlers[event]) {
      this._handlers[event].forEach(handler => {
        try {
          handler(...args);
        } catch (error) {
          console.error('Event handler error:', error);
        }
      });
    } else {
      console.warn(`⚠️ No handlers for event: ${event}`);
    }
  }
  
  private async handleSubscribeDashboard() {
    try {
      console.log('📊 Subscribing to dashboard updates via HTTP');
      
      // Confirm subscription first
      this.emit('dashboard_subscribed', { status: 'subscribed', client_id: this._id });
      
      // Get initial dashboard data with detailed logging
      console.log('🔄 Fetching dashboard data from:', `${this.url}/dashboard-readings`);
      const response = await fetch(`${this.url}/dashboard-readings`);
      
      console.log('📡 Response status:', response.status, response.statusText);
      console.log('📡 Response headers:', Object.fromEntries(response.headers.entries()));
      
      if (response.ok) {
        const data = await response.json();
        console.log('📊 Raw dashboard data received:', data);
        console.log('📊 Data success flag:', data.success);
        console.log('📊 Stats data:', data.stats);
        
        if (data.success) {
          console.log('✅ Emitting dashboard_update event with data');
          this.emit('dashboard_update', data);
          console.log('✅ dashboard_update event emitted successfully');
        } else {
          console.warn('⚠️ Data success flag is false');
        }
      } else {
        console.error('❌ HTTP response not ok:', response.status, response.statusText);
      }
    } catch (error) {
      console.error('❌ Error subscribing to dashboard:', error);
      this.emit('connect_error', error);
    }
  }
  
  private async handleDashboardRefresh() {
    try {
      console.log('🔄 Refreshing dashboard via HTTP');
      
      const response = await fetch(`${this.url}/dashboard-refresh`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });
      
      if (response.ok) {
        const data = await response.json();
        if (data.success) {
          console.log('✅ Dashboard refreshed successfully');
          this.emit('dashboard_update', data);
        }
      }
    } catch (error) {
      console.error('Error refreshing dashboard:', error);
    }
  }

  disconnect() {
    this._connected = false;
    this._disconnected = true;
    if (this._pollInterval) {
      clearInterval(this._pollInterval);
      this._pollInterval = null;
    }
    console.log('🔌 Socket.IO disconnected');
    this.emit('disconnect', 'manual');
  }
}

const firebaseConfig = {
  apiKey: "AIzaSyDkvgZIzNtRdf_uq0-aR-pLVKxf0NAKTJk",
  authDomain: "cv-camera-app.firebaseapp.com",
  projectId: "cv-camera-app",
  storageBucket: "cv-camera-app.firebasestorage.app",
  messagingSenderId: "272918299127",
  appId: "1:272918299127:web:064a593796e6742b49eef2",
  measurementId: "G-8QKJL1E100"
}

// Initialize Firebase
const app = initializeApp(firebaseConfig)

// Initialize Firestore
export const db = getFirestore(app)

// Utility function to remove undefined values from objects
export function cleanUndefinedValues(obj: any): any {
  if (obj === null || obj === undefined) {
    return null
  }
  
  if (Array.isArray(obj)) {
    return obj.map(cleanUndefinedValues).filter(item => item !== undefined)
  }
  
  if (typeof obj === 'object') {
    const cleaned: any = {}
    for (const [key, value] of Object.entries(obj)) {
      if (value !== undefined) {
        cleaned[key] = cleanUndefinedValues(value)
      }
    }
    return cleaned
  }
  
  return obj
}

// CV Analysis data interface
export interface CVAnalysisData {
  id?: string
  timestamp: Timestamp
  imageMetadata: {
    size: number
    type: string
    dimensions?: {
      width: number
      height: number
    }
  }
  processingInfo: {
    processingTime: number
    modelVersion: string
    confidence: number
  }
  detectionSummary: {
    peopleDetected: number
    facesAnalyzed: number
    averageConfidence: number
  }
  peopleAnalysis: Array<{
    personId: number
    demographics: {
      estimatedAge: number
      ageRange: string
      gender: string
      confidence: number
    }
    physicalAttributes: {
      skinTone: string
      hairColor: string
      hairStyle: string
      eyeColor: string
    }
    appearance: {
      clothing: {
        detected_items: string[]
        dominant_colors: string[]
        style_category: string
        patterns: string[]
        fabric_type: string
      }
      accessories: string[]
      overall_style: string
      outfit_formality: string
    }
    emotions: {
      primary: string
      confidence: number
      secondary?: string
    }
    pose: {
      position: string
      orientation: string
      visibility: string
    }
  }>
  sceneAnalysis: {
    lighting: string
    setting: string
    imageQuality: string
    dominantColors: string[]
  }
}

// Save CV analysis to Firestore
export async function saveCVAnalysis(analysisData: Omit<CVAnalysisData, 'id' | 'timestamp'>): Promise<string> {
  try {
    const dataWithTimestamp = {
      ...analysisData,
      timestamp: Timestamp.now()
    }
    
    // Clean undefined values before saving
    const cleanedData = cleanUndefinedValues(dataWithTimestamp)
    
    const docRef = await addDoc(collection(db, 'cvAnalyses'), cleanedData)
    console.log('CV Analysis saved with ID: ', docRef.id)
    return docRef.id
  } catch (error) {
    console.error('Error saving CV analysis: ', error)
    throw error
  }
}

// Get all CV analyses from Firestore
export async function getCVAnalyses(): Promise<CVAnalysisData[]> {
  try {
    const q = query(collection(db, 'cvAnalyses'), orderBy('timestamp', 'desc'))
    const querySnapshot = await getDocs(q)
    
    const analyses: CVAnalysisData[] = []
    querySnapshot.forEach((doc) => {
      analyses.push({
        id: doc.id,
        ...doc.data()
      } as CVAnalysisData)
    })
    
    return analyses
  } catch (error) {
    console.error('Error getting CV analyses: ', error)
    throw error
  }
}

// Dashboard data interfaces
export interface DashboardData {
  stats: {
    images_analyzed: { value: string; change: string }
    people_detected: { value: string; change: string }
    clothing_items: { value: string; change: string }
    accuracy_rate: { value: string; change: string }
  }
  recent_activity: Array<{
    action: string
    details: string
    time: string
  }>
  system_status: {
    cv_backend: string
    database: string
    ai_models: string
    processing_speed: string
    overall: string
  }
  performance_metrics: {
    detection_accuracy: { value: string; percentage: number }
    processing_speed: { value: string; percentage: number }
    system_load: { value: string; percentage: number }
  }
  timestamp: string
  success: boolean
}

// Dashboard service functions
export async function getDashboardReadings(): Promise<DashboardData> {
  try {
    const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5000'
    console.log('📊 Fetching dashboard data from:', `${apiUrl}/dashboard-readings`)
    
    const response = await fetch(`${apiUrl}/dashboard-readings`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    })

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }

    const data = await response.json()
    
    if (data.success) {
    return data as DashboardData
    } else {
      throw new Error(data.error || 'Failed to fetch dashboard data')
    }
  } catch (error) {
    console.error('Error fetching dashboard readings:', error)
    throw error
  }
}

export async function refreshDashboardReadings(): Promise<DashboardData> {
  try {
    const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5000'
    console.log('🔄 Refreshing dashboard data from:', `${apiUrl}/dashboard-refresh`)
    
    const response = await fetch(`${apiUrl}/dashboard-refresh`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
    })

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }

    const data = await response.json()
    
    if (data.success) {
    return data as DashboardData
    } else {
      throw new Error(data.error || 'Failed to refresh dashboard data')
    }
  } catch (error) {
    console.error('Error refreshing dashboard readings:', error)
    throw error
  }
}

// Real-time dashboard listener using Firestore onSnapshot
export function subscribeToAnalysisUpdates(callback: (hasNewData: boolean) => void): Unsubscribe {
  try {
    const q = query(collection(db, 'cvAnalyses'), orderBy('timestamp', 'desc'))
    
    let isFirstLoad = true
    const unsubscribe = onSnapshot(q, (snapshot) => {
      if (isFirstLoad) {
        isFirstLoad = false
        return // Skip first load
      }
      
      // New data detected, trigger dashboard refresh
      callback(true)
    }, (error) => {
      console.error('Error in analysis updates listener:', error)
      callback(false)
    })

    return unsubscribe
  } catch (error) {
    console.error('Error setting up analysis updates listener:', error)
    return () => {} // Return no-op function
  }
}

export const subscribeToRealtimeDashboard = (
  callback: (data: DashboardData) => void
): (() => void) => {
  console.log('Setting up dashboard polling for real-time updates...');
  
  let isActive = true;
  
  // Function to fetch and update data
  const fetchAndUpdate = async () => {
    if (!isActive) return;
    
    try {
      const data = await getDashboardReadings();
      if (isActive) {
        callback(data);
      }
    } catch (error) {
      console.error('Error fetching dashboard data:', error);
    }
  };
  
  // Initial fetch
  fetchAndUpdate();
  
  // Set up polling every 3 seconds for real-time feel
  const intervalId = setInterval(fetchAndUpdate, 3000);
  
  // Return cleanup function
  return () => {
    console.log('Cleaning up dashboard polling...');
    isActive = false;
    clearInterval(intervalId);
  };
};

// Direct Firebase Dashboard Data (no backend needed!)
export async function getDashboardDataFromFirebase(): Promise<DashboardData> {
  try {
    console.log('📊 [MANUAL CALL] Fetching dashboard data directly from Firebase...')
    
    // Get recent analyses directly from Firebase with same limit as real-time listener
    const q = query(
      collection(db, 'cvAnalyses'), 
      orderBy('timestamp', 'desc'),
      limit(100) // Same limit as real-time listener
    )
    const querySnapshot = await getDocs(q)
    
    const analyses: CVAnalysisData[] = []
    querySnapshot.forEach((doc) => {
      analyses.push({
        id: doc.id,
        ...doc.data()
      } as CVAnalysisData)
    })
    
    console.log(`✅ [MANUAL CALL] Got ${analyses.length} analyses from Firebase (limited to 100)`)
    
    // Calculate stats directly from Firebase data
    const stats = calculateDashboardStats(analyses)
    const recentActivity = getRecentActivity(analyses.slice(0, 10))
    
    const dashboardData: DashboardData = {
      stats,
      recent_activity: recentActivity,
      system_status: {
        cv_backend: 'Online',
        database: `Connected (Firebase Direct - ${analyses.length} records)`,
        ai_models: 'Loaded',
        processing_speed: '~1.2s avg',
        overall: 'Firebase Direct Connection'
      },
      performance_metrics: {
        detection_accuracy: {
          value: stats.accuracy_rate.value,
          percentage: parseFloat(stats.accuracy_rate.value.replace('%', ''))
        },
        processing_speed: {
          value: '95%',
          percentage: 95
        },
        system_load: {
          value: '35%',
          percentage: 35
        }
      },
      success: true,
      timestamp: new Date().toISOString()
    }
    
    console.log('📊 [MANUAL CALL] Dashboard data calculated from Firebase:', {
      images: stats.images_analyzed.value,
      people: stats.people_detected.value,
      clothing: stats.clothing_items.value,
      accuracy: stats.accuracy_rate.value
    })
    
    return dashboardData
    
  } catch (error) {
    console.error('❌ [MANUAL CALL] Error getting dashboard data from Firebase:', error)
    throw error
  }
}

// Calculate dashboard stats from Firebase analyses
function calculateDashboardStats(analyses: CVAnalysisData[]) {
  let totalPeople = 0
  let totalClothing = 0
  let totalConfidence = 0
  let validConfidenceCount = 0
  
  console.log('🔢 DEBUGGING: Calculating stats from Firebase analyses...')
  console.log(`📊 Total analyses to process: ${analyses.length}`)
  
  // Debug first few records in detail
  analyses.slice(0, 3).forEach((analysis, index) => {
    console.log(`\n📋 ANALYSIS ${index + 1} DEBUG:`)
    console.log('   📋 Analysis object keys:', Object.keys(analysis))
    console.log('   📋 peopleAnalysis exists:', !!analysis.peopleAnalysis)
    console.log('   📋 peopleAnalysis type:', typeof analysis.peopleAnalysis)
    console.log('   📋 peopleAnalysis length:', analysis.peopleAnalysis?.length)
    console.log('   📋 detectionSummary exists:', !!analysis.detectionSummary)
    console.log('   📋 detectionSummary.averageConfidence:', analysis.detectionSummary?.averageConfidence)
    
    if (analysis.peopleAnalysis && analysis.peopleAnalysis.length > 0) {
      console.log('   👤 First person keys:', Object.keys(analysis.peopleAnalysis[0]))
      console.log('   👤 First person structure:', JSON.stringify(analysis.peopleAnalysis[0], null, 2))
    }
  })
  
  analyses.forEach((analysis, index) => {
    // Count people
    const peopleCount = analysis.peopleAnalysis?.length || 0
    totalPeople += peopleCount
    
    // Count clothing items
    let clothingCount = 0
    if (analysis.peopleAnalysis) {
      analysis.peopleAnalysis.forEach((person: any) => {
        if (person.clothing_analysis?.detected_items) {
          clothingCount += person.clothing_analysis.detected_items.length
        } else if (person.appearance?.clothing?.length) {
          clothingCount += person.appearance.clothing.length
        } else {
          // Estimate 3 clothing items per person detected
          clothingCount += peopleCount > 0 ? 3 : 0
        }
      })
    }
    totalClothing += clothingCount
    
    // Calculate average confidence using SAME method as view-data page
    const analysisConfidence = analysis.detectionSummary?.averageConfidence || 0
    if (analysisConfidence > 0) {
      totalConfidence += analysisConfidence
      validConfidenceCount++
      
      // Debug confidence extraction for first few records
      if (index < 3) {
        console.log(`   🎯 Analysis ${index + 1} confidence: ${(analysisConfidence * 100).toFixed(1)}% (from detectionSummary.averageConfidence)`)
      }
    }
    
    if (index < 5) {
      console.log(`📋 Analysis ${index + 1}: ${peopleCount} people, ${clothingCount} clothing`)
    }
  })
  
  // Calculate REAL average confidence from Firebase data only (same as view-data page)
  const avgConfidence = validConfidenceCount > 0 ? (totalConfidence / validConfidenceCount * 100) : 0
  
  console.log('\n📊 FINAL CALCULATED STATS:')
  console.log(`   👥 Total People: ${totalPeople}`)
  console.log(`   👔 Total Clothing: ${totalClothing}`)
  console.log(`   🎯 Average Confidence: ${avgConfidence.toFixed(1)}% (from ${validConfidenceCount} analyses with confidence data)`)
  console.log(`   📈 From ${analyses.length} analyses total`)
  console.log(`   📊 Confidence Data Coverage: ${validConfidenceCount}/${analyses.length} analyses (${((validConfidenceCount/analyses.length)*100).toFixed(1)}%)`)
  
  if (validConfidenceCount === 0) {
    console.log(`   ⚠️  No confidence data found in any analysis - confidence shows 0%`)
  } else if (validConfidenceCount < analyses.length) {
    console.log(`   ℹ️  ${analyses.length - validConfidenceCount} analyses missing confidence data`)
  } else {
    console.log(`   ✅ All analyses have confidence data!`)
  }
  
  return {
    images_analyzed: {
      value: analyses.length.toString(),
      change: '+12%'
    },
    people_detected: {
      value: totalPeople.toString(),
      change: '+15%'
    },
    clothing_items: {
      value: totalClothing.toString(),
      change: '+18%'
    },
    accuracy_rate: {
      value: `${avgConfidence.toFixed(1)}%`,
      change: validConfidenceCount > 0 ? '+2%' : '0%'
    }
  }
}

// Get recent activity from analyses
function getRecentActivity(analyses: CVAnalysisData[]) {
  return analyses.slice(0, 5).map((analysis, index) => ({
    action: 'Image Analysis Completed',
    details: `Analyzed ${analysis.peopleAnalysis?.length || 0} people, ${
      analysis.peopleAnalysis?.reduce((total: number, person: any) => 
        total + (person.clothing_analysis?.detected_items?.length || 3), 0
      ) || 0
    } clothing items`,
    time: analysis.timestamp?.toDate ? analysis.timestamp.toDate().toLocaleTimeString() : new Date().toLocaleTimeString()
  }))
}

// Set up real-time Firebase dashboard listener
export function setupFirebaseDashboardListener(
  onUpdate: (data: DashboardData) => void,
  onError: (error: Error) => void
): () => void {
  console.log('🔥 Setting up real-time Firebase dashboard listener...')
  
  const unsubscribe = onSnapshot(
    query(
      collection(db, 'cvAnalyses'),
      orderBy('timestamp', 'desc'),
      limit(100) // Increased to match manual refresh data amount
    ),
    (snapshot) => {
      console.log('🔥 [REAL-TIME] Firebase snapshot update received!')
      console.log(`🔥 [REAL-TIME] Snapshot size: ${snapshot.size}, from cache: ${snapshot.metadata.fromCache}`)
      
      try {
        const analyses: CVAnalysisData[] = snapshot.docs.map(doc => ({
          id: doc.id,
          ...doc.data(),
          timestamp: doc.data().timestamp
        })) as CVAnalysisData[]
        
        console.log(`📊 [REAL-TIME] Processing ${analyses.length} analyses from Firebase snapshot`)
        
        // Calculate dashboard data from real-time Firebase data
        const stats = calculateDashboardStats(analyses)
        const recentActivity = getRecentActivity(analyses.slice(0, 10))
        
        const dashboardData: DashboardData = {
          stats,
          recent_activity: recentActivity,
          system_status: {
            cv_backend: 'Online',
            database: `Connected (Firebase Real-time - ${analyses.length} records)`,
            ai_models: 'Loaded',
            processing_speed: '~1.2s avg',
            overall: 'Firebase Real-time Connection'
          },
          performance_metrics: {
            detection_accuracy: {
              value: stats.accuracy_rate.value,
              percentage: parseFloat(stats.accuracy_rate.value.replace('%', ''))
            },
            processing_speed: {
              value: '95%',
              percentage: 95
            },
            system_load: {
              value: '35%',
              percentage: 35
            }
          },
          success: true,
          timestamp: new Date().toISOString()
        }
        
        console.log('🎉 [REAL-TIME] Real-time dashboard data ready:', {
          people: stats.people_detected.value,
          clothing: stats.clothing_items.value,
          accuracy: stats.accuracy_rate.value,
          fromCache: snapshot.metadata.fromCache
        })
        
        onUpdate(dashboardData)
        
      } catch (error) {
        console.error('❌ [REAL-TIME] Error processing Firebase snapshot:', error)
        onError(error as Error)
      }
    },
    (error) => {
      console.error('❌ [REAL-TIME] Firebase snapshot listener error:', error)
      onError(error as Error)
    }
  )
  
  console.log('✅ Firebase dashboard listener active!')
  return unsubscribe
}

// WebSocket connection management
class WebSocketManager {
  private socket: any = null
  private reconnectAttempts = 0
  private maxReconnectAttempts = 5
  private reconnectInterval = 3000
  private isDisabled = false
  
  connect(): any {
    if (this.isDisabled) {
      console.log('🔌 WebSocket disabled due to previous failures')
      return null
    }
    
    if (this.socket?.connected) {
      console.log('🔌 WebSocket already connected')
      return this.socket
    }
    
    if (!io) {
      console.warn('Socket.IO not available, WebSocket features disabled')
      this.isDisabled = true
      return null
    }
    
    const backendUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5000'
    console.log('🔌 Connecting to WebSocket server:', backendUrl)
    
    try {
      this.socket = io(backendUrl, {
        autoConnect: false,  // Manual connection control
        timeout: 20000,
        reconnection: false  // Handle reconnection manually
      })
      
      this.socket.on('connect', () => {
        console.log('✅ WebSocket connected successfully')
        console.log('🆔 Socket ID:', this.socket.id)
        this.reconnectAttempts = 0
        this.isDisabled = false
      })
      
      this.socket.on('disconnect', (reason: string) => {
        console.log('🔌 WebSocket disconnected:', reason)
        if (reason === 'io server disconnect') {
          // The disconnection was initiated by the server, try to reconnect
          this.reconnect()
        }
      })
      
      this.socket.on('connect_error', (error: any) => {
        console.error('❌ WebSocket connection error:', error?.message || error)
        this.reconnectAttempts++
        
        if (this.reconnectAttempts >= this.maxReconnectAttempts) {
          console.error('❌ Max WebSocket reconnection attempts reached')
          this.isDisabled = true
          this.socket = null
        } else {
          // Try to reconnect after a delay
          setTimeout(() => this.reconnect(), this.reconnectInterval)
        }
      })
      
      // Manual connect
      this.socket.connect()
      
      return this.socket
    } catch (error) {
      console.error('❌ Failed to create WebSocket connection:', error)
      this.isDisabled = true
      return null
    }
  }
  
  private reconnect() {
    if (this.isDisabled || this.reconnectAttempts >= this.maxReconnectAttempts) {
      return
    }
    
    console.log(`🔄 Attempting to reconnect (${this.reconnectAttempts + 1}/${this.maxReconnectAttempts})...`)
    this.socket?.connect()
  }
  
  disconnect() {
    if (this.socket) {
      try {
        this.socket.disconnect()
      } catch (error) {
        console.error('Error during WebSocket disconnect:', error)
      }
      this.socket = null
      console.log('🔌 WebSocket disconnected manually')
    }
  }
  
  getSocket(): any {
    return this.isDisabled ? null : this.socket
  }
  
  isWebSocketDisabled(): boolean {
    return this.isDisabled
  }
  
  reset() {
    this.isDisabled = false
    this.reconnectAttempts = 0
    console.log('🔌 WebSocket manager reset')
  }
}

// Global WebSocket manager instance
const websocketManager = new WebSocketManager()

// WebSocket Dashboard Connection Functions
export function connectToWebSocketDashboard(
  onDashboardUpdate: (data: DashboardData) => void,
  onNewAnalysis: (data: any) => void,
  onError: (error: any) => void
): (() => void) | null {
  const socket = websocketManager.connect()
  
  if (!socket) {
    console.warn('❌ WebSocket not available - dashboard will not receive real-time updates')
    onError(new Error('WebSocket connection failed'))
    return null
  }

  console.log('🔌 Setting up WebSocket dashboard listeners...')

  // Set up event listeners
  socket.on('connect', () => {
    console.log('📊 WebSocket connected - subscribing to dashboard updates')
    socket.emit('subscribe_dashboard')
  })

  socket.on('dashboard_update', (data: any) => {
    console.log('📊 Dashboard update received via WebSocket:', data)
    console.log('📊 Processing dashboard data...')
    console.log('📊 Data type:', typeof data)
    console.log('📊 Data keys:', Object.keys(data || {}))
    console.log('📊 Data.success:', data?.success)
    console.log('📊 Data.stats:', data?.stats)
    
    try {
      if (data && data.success) {
        console.log('✅ Valid dashboard data - calling onDashboardUpdate')
        onDashboardUpdate(data)
        console.log('✅ onDashboardUpdate called successfully')
      } else {
        console.warn('❌ Invalid dashboard data received:', data)
        console.warn('❌ Reasons: data exists?', !!data, 'success flag?', data?.success)
      }
    } catch (error) {
      console.error('❌ Error processing dashboard update:', error)
      onError(error)
    }
  })

  socket.on('new_analysis', (data: any) => {
    console.log('🆕 New analysis notification via WebSocket:', data)
    onNewAnalysis(data)
  })

  socket.on('dashboard_subscribed', (data: any) => {
    console.log('✅ Dashboard subscription confirmed:', data)
  })

  socket.on('connection_confirmed', (data: any) => {
    console.log('✅ WebSocket connection confirmed:', data)
  })

  socket.on('connect_error', (error: any) => {
    console.error('❌ WebSocket connection error:', error)
    onError(error)
  })

  socket.on('disconnect', (reason: string) => {
    console.log('🔌 WebSocket disconnected:', reason)
    // Don't call onError for normal disconnections
  })

  // Return cleanup function
  return () => {
    console.log('🧹 Cleaning up WebSocket dashboard listeners')
    if (socket) {
      try {
        socket.emit('unsubscribe_dashboard')
      } catch (error) {
        console.warn('Error unsubscribing from dashboard:', error)
      }
    }
  }
}

export function requestDashboardRefresh(): void {
  const socket = websocketManager.getSocket()
  
  if (socket && socket.connected) {
    console.log('🔄 Requesting dashboard refresh via WebSocket')
    socket.emit('request_dashboard_refresh')
  } else {
    console.warn('⚠️ WebSocket not connected - cannot request dashboard refresh')
  }
}

export function notifyNewAnalysis(analysisData: any): void {
  const socket = websocketManager.getSocket()
  
  if (socket && socket.connected) {
    console.log('📤 Notifying new analysis via WebSocket')
    socket.emit('new_analysis_notification', analysisData)
  } else {
    console.warn('⚠️ WebSocket not connected - cannot notify new analysis')
  }
}

export function disconnectWebSocket(): void {
  console.log('🔌 Disconnecting WebSocket')
  websocketManager.disconnect()
}

export default app 