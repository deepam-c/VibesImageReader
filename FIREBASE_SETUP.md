# 🔥 Firebase Setup Guide

This guide will help you set up Firebase for storing your CV analysis data.

## Step 1: Create Firebase Project

1. Go to [Firebase Console](https://console.firebase.google.com/)
2. Click "Create a project"
3. Enter a project name (e.g., "cv-camera-app")
4. Disable Google Analytics (optional for this project)
5. Click "Create project"

## Step 2: Enable Firestore Database

1. In your Firebase project, click "Firestore Database" in the left sidebar
2. Click "Create database"
3. Choose "Start in test mode" (for development)
4. Select a location close to you
5. Click "Done"

## Step 3: Get Configuration Keys

1. Go to Project Settings (gear icon in sidebar)
2. Scroll to "Your apps" section
3. Click the "Web" icon (`</>`)
4. Register your app with a nickname (e.g., "CV Camera")
5. Copy the `firebaseConfig` object

## Step 4: Update Your Project

1. Open `lib/firebase.ts` in your project
2. Replace the placeholder config with your actual config:

```typescript
const firebaseConfig = {
  apiKey: "your-actual-api-key-here",
  authDomain: "your-project-id.firebaseapp.com",
  projectId: "your-actual-project-id",
  storageBucket: "your-project-id.appspot.com",
  messagingSenderId: "123456789012",
  appId: "1:123456789012:web:abcdef123456"
}
```

## Step 5: Test Connection

1. Start your application (`npm run dev`)
2. Capture or upload an image
3. Check the browser console for "Analysis saved to Firebase with ID: ..."
4. Go to Firebase Console > Firestore Database to see your saved data

## Firestore Data Structure

Your CV analyses will be saved in the `cvAnalyses` collection with this structure:

```
cvAnalyses/
├── {document-id}/
│   ├── timestamp: Date
│   ├── imageMetadata: Object
│   ├── processingInfo: Object
│   ├── detectionSummary: Object
│   ├── peopleAnalysis: Array
│   └── sceneAnalysis: Object
```

## Optional: Production Security

For production use, update your Firestore security rules:

```javascript
rules_version = '2';
service cloud.firestore {
  match /databases/{database}/documents {
    match /cvAnalyses/{document} {
      // Add authentication rules here
      allow read, write: if request.auth != null;
    }
  }
}
```

## Troubleshooting

- **Permission denied**: Check Firestore rules allow read/write
- **Project not found**: Verify project ID in config
- **Invalid API key**: Regenerate keys in Firebase console
- **Network error**: Check internet connection and Firebase service status

That's it! Your CV analysis data will now be automatically saved to Firebase. 🎉 