# Computer Vision Camera Application

A powerful Next.js application with advanced computer vision capabilities for analyzing human images, detecting clothing, accessories, and personal attributes.

## 🌟 Features

### 📸 **Image Capture & Upload**
- Real-time camera capture
- Image upload from device
- Instant CV analysis processing

### 🧠 **Advanced Computer Vision Analysis**
- **Person Detection**: Accurate human detection and face analysis
- **Demographics**: Age estimation, gender detection, emotion analysis
- **Clothing Detection**: 
  - Specific items (shirts, jackets, pants, accessories)
  - Fabric types and patterns (stripes, solid, plaid)
  - Style categorization (formal, casual, business)
- **Accessory Recognition**:
  - Jewelry (earrings, necklaces, bracelets, watches)
  - Headwear (caps, hats, headbands)
  - Eyewear (glasses, sunglasses)
  - Other accessories (ties, scarves, bags)

### 🔥 **Firebase Integration**
- **NoSQL Data Storage**: All CV analysis results saved to Firestore
- **Real-time Data**: Instant saving and retrieval
- **Data Persistence**: Historical analysis tracking

### 📊 **Data Management**
- **View Data Page**: Browse all saved analyses
- **Search & Filter**: Find specific analyses by style, formality, etc.
- **JSON Export**: Copy raw analysis data
- **Detailed View**: Comprehensive analysis breakdown

## 🚀 Quick Start

### Prerequisites
- Node.js 18+ 
- Python 3.8+
- Firebase project (for data storage)

### 1. Frontend Setup
```bash
# Install dependencies
npm install

# Start development server
npm run dev
```

### 2. Backend Setup
```bash
# Navigate to backend
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Start the CV backend
python app_smart.py
```

### 3. Firebase Configuration

1. **Create a Firebase Project**:
   - Go to [Firebase Console](https://console.firebase.google.com/)
   - Create a new project
   - Enable Firestore Database

2. **Get Firebase Config**:
   - Go to Project Settings > General
   - Scroll to "Your apps" section
   - Copy the firebaseConfig object

3. **Update Configuration**:
   - Edit `lib/firebase.ts`
   - Replace the placeholder config with your Firebase project config:
   ```typescript
   const firebaseConfig = {
     apiKey: "your-actual-api-key",
     authDomain: "your-project.firebaseapp.com",
     projectId: "your-actual-project-id",
     storageBucket: "your-project.appspot.com",
     messagingSenderId: "your-actual-sender-id",
     appId: "your-actual-app-id"
   }
   ```

4. **Firestore Rules** (Optional - for production):
   ```javascript
   rules_version = '2';
   service cloud.firestore {
     match /databases/{database}/documents {
       match /cvAnalyses/{document} {
         allow read, write: if true; // Configure based on your security needs
       }
     }
   }
   ```

## 📱 Usage

### 1. **Capture Analysis**
- Navigate to `/capture`
- Use "Start Camera" for live capture OR "Upload Image" for file upload
- View instant CV analysis results
- Data automatically saved to Firebase

### 2. **View Saved Data**
- Navigate to `/view-data`
- Browse all historical analyses
- Use search and filters to find specific data
- Click on any analysis for detailed view
- Copy JSON data for external use

## 🏗️ Application Structure

```
├── app/
│   ├── capture/          # Camera capture page
│   ├── view-data/        # Data viewing page
│   └── layout.tsx        # App layout
├── components/
│   ├── CameraCapture.tsx # Camera & upload component
│   ├── AnalysisResults.tsx # CV results display
│   └── Sidebar.tsx       # Navigation
├── lib/
│   └── firebase.ts       # Firebase configuration
└── backend/
    ├── app_smart.py      # Enhanced CV backend
    ├── requirements.txt  # Python dependencies
    └── utils/            # CV processing utilities
```

## 🔧 Configuration

### Environment Variables
Create `.env.local` (optional):
```
NEXT_PUBLIC_CV_BACKEND_URL=http://localhost:5000
```

### Backend Configuration
The CV backend runs on `http://localhost:5000` by default. Modify `backend/app_smart.py` to change settings.

## 📡 API Endpoints

### CV Backend (`localhost:5000`)
- `GET /health` - Health check
- `POST /analyze-image` - Analyze uploaded image
- `GET /get-capabilities` - Get CV model capabilities

### Firebase Integration
- Automatic saving to `cvAnalyses` collection
- Real-time data synchronization
- Structured NoSQL format for analysis results

## 🎨 CV Analysis Output

Each analysis includes:
```json
{
  "timestamp": "2024-01-20T10:30:00Z",
  "imageMetadata": {
    "size": 245760,
    "type": "image/jpeg"
  },
  "detectionSummary": {
    "peopleDetected": 1,
    "facesAnalyzed": 1,
    "averageConfidence": 0.87
  },
  "peopleAnalysis": [{
    "demographics": {
      "estimatedAge": 28,
      "gender": "female",
      "confidence": 0.89
    },
    "appearance": {
      "clothing": {
        "detected_items": ["blouse", "jeans"],
        "patterns": ["solid"],
        "style_category": "casual"
      },
      "accessories": ["earrings", "watch"],
      "outfit_formality": "smart_casual"
    },
    "emotions": {
      "primary": "happy",
      "confidence": 0.76
    }
  }]
}
```

## 🛠️ Troubleshooting

### Common Issues

1. **Firebase Connection Error**:
   - Verify your Firebase config in `lib/firebase.ts`
   - Check Firestore rules allow read/write
   - Ensure Firebase project is active

2. **Backend Not Running**:
   - Ensure virtual environment is activated
   - Check Python dependencies are installed
   - Verify port 5000 is available

3. **Camera Access Denied**:
   - Enable camera permissions in browser
   - Use HTTPS in production
   - Check browser compatibility

## 📄 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create Pull Request

## 📞 Support

For issues and questions:
- Check the troubleshooting section
- Review Firebase and OpenCV documentation
- Open an issue on GitHub 