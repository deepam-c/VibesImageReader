import { initializeApp } from 'firebase/app'
import { getFirestore, collection, addDoc, getDocs, orderBy, query, Timestamp } from 'firebase/firestore'

const
 firebaseConfig =
{
  apiKey:
"AIzaSyDkvgZIzNtRdf_uq0-aR-pLVKxf0NAKTJk",
  authDomain:"cv-camera-app.firebaseapp.com",
  projectId:"cv-camera-app",
  storageBucket:"cv-camera-app.firebasestorage.app",
  messagingSenderId:"272918299127",
  appId:"1:272918299127:web:064a593796e6742b49eef2",
  measurementId:"G-8QKJL1E100"
};

/*const firebaseConfig = {
  // Replace with your Firebase project configuration
  apiKey: "your-api-key",
  authDomain: "your-project.firebaseapp.com",
  projectId: "your-project-id",
  storageBucket: "your-project.appspot.com",
  messagingSenderId: "123456789",
  appId: "your-app-id"
}*/

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

export default app 