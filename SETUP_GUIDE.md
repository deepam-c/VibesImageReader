# Complete Setup Guide - Camera Capture with Computer Vision Analysis

This project combines a Next.js camera capture frontend with a powerful Python computer vision backend for comprehensive human analysis.

## 🚀 **Project Overview**

### **Frontend (Next.js)**
- Camera capture interface
- Real-time image analysis display
- Modern UI with Tailwind CSS
- TypeScript for type safety

### **Backend (Python)**
- Advanced computer vision analysis
- Multi-model AI pipeline
- RESTful API endpoints
- Detailed person analysis including:
  - Demographics (age, gender)
  - Physical attributes (hair, facial features)
  - Clothing and style analysis
  - Pose and activity detection

---

## 📋 **Prerequisites**

### **System Requirements**
- **Node.js** 18+ (for frontend)
- **Python** 3.8+ (for backend)
- **2-4GB RAM** (for AI models)
- **2GB disk space** (for model weights)
- **Camera** (webcam or device camera)

### **Optional but Recommended**
- **CUDA-capable GPU** (for faster processing)
- **Git** (for version control)

---

## 🔧 **Installation Steps**

### **Step 1: Setup Frontend (Next.js)**

1. **Install Dependencies**
   ```bash
   npm install
   ```

2. **Start Development Server**
   ```bash
   npm run dev
   ```
   
   ✅ Frontend will be available at: `http://localhost:3001`

### **Step 2: Setup Backend (Python)**

1. **Navigate to Backend Directory**
   ```bash
   cd backend
   ```

2. **Create Virtual Environment**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate
   
   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Python Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   
   > ⏱️ **Note**: This may take 5-10 minutes as it downloads large AI models

4. **Start Backend Server**
   ```bash
   python app.py
   ```
   
   ✅ Backend API will be available at: `http://localhost:5000`

---

## 🎯 **Quick Start Usage**

1. **Ensure both servers are running:**
   - Frontend: `http://localhost:3001`
   - Backend: `http://localhost:5000`

2. **Access the application:**
   - Open your browser to `http://localhost:3001`
   - Navigate to `/capture` page

3. **Start using camera capture:**
   - Click "Start Camera"
   - Allow camera permissions
   - Click "Capture Image"
   - Wait for AI analysis results

---

## 🔍 **Features Overview**

### **Computer Vision Analysis Includes:**

#### **📊 Demographics**
- Age estimation with confidence scores
- Gender detection
- Age range categorization

#### **👤 Physical Attributes**
- Facial feature analysis
- Hair style, color, and length detection
- Body build estimation
- Visible body parts identification

#### **👕 Appearance Analysis**
- Clothing item detection
- Color palette analysis
- Style classification
- Accessory identification

#### **🤸 Pose & Activity**
- Real-time pose estimation
- Body position detection
- Activity inference
- Confidence metrics

#### **📈 Scene Analysis**
- Multi-person detection
- Image quality assessment
- Lighting condition analysis
- Processing statistics

---

## 🛠️ **Configuration Options**

### **Frontend Configuration**
```typescript
// In components/CameraCapture.tsx
const BACKEND_URL = 'http://localhost:5000' // Change if backend runs elsewhere
```

### **Backend Configuration**
```python
# In backend/app.py
app.run(debug=True, host='0.0.0.0', port=5000)
```

---

## 🧪 **Testing the System**

### **1. Health Check**
```bash
curl http://localhost:5000/health
```

### **2. Test Image Analysis**
```bash
curl -X POST http://localhost:5000/analyze-image \
  -H "Content-Type: application/json" \
  -d '{"image": "data:image/jpeg;base64,..."}'
```

### **3. Check Capabilities**
```bash
curl http://localhost:5000/get-capabilities
```

---

## 📱 **API Endpoints**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Server health check |
| `/analyze-image` | POST | Analyze uploaded image |
| `/analyze-video-frame` | POST | Analyze video frame |
| `/get-capabilities` | GET | List system capabilities |

---

## ⚡ **Performance Optimization**

### **For CPU-Only Systems:**
- Analysis time: 2-5 seconds per image
- Memory usage: ~2-3GB
- Recommended image size: 1024x768

### **For GPU-Enabled Systems:**
1. **Install CUDA version of TensorFlow:**
   ```bash
   pip install tensorflow-gpu==2.15.0
   ```

2. **Enable GPU in environment:**
   ```bash
   echo "ENABLE_GPU=true" >> .env
   ```

- Analysis time: 0.5-1 second per image
- Better for real-time processing

---

## 🚨 **Troubleshooting**

### **Common Issues & Solutions**

#### **"Module not found" errors:**
```bash
# Ensure virtual environment is activated
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# Reinstall dependencies
pip install -r requirements.txt
```

#### **Camera permissions denied:**
- Allow camera access in browser settings
- Check system camera permissions
- Try HTTPS for production deployments

#### **Backend connection errors:**
- Verify backend is running on port 5000
- Check firewall settings
- Ensure CORS is properly configured

#### **Memory issues:**
```bash
# Reduce image processing size
# Edit backend/utils/image_processor.py
self.resize_image(image, max_width=512, max_height=512)
```

#### **Model download failures:**
```bash
# Manual model download
python -c "from deepface import DeepFace; DeepFace.build_model('Age')"
python -c "from deepface import DeepFace; DeepFace.build_model('Gender')"
```

---

## 🔒 **Security Considerations**

### **Development**
- Both servers run on localhost
- CORS enabled for local development
- Image data processed in memory only

### **Production Deployment**
- Use HTTPS for both frontend and backend
- Implement authentication for API endpoints
- Add rate limiting and input validation
- Consider image size limits
- Set up proper logging and monitoring

---

## 📦 **Project Structure**

```
Proj-1/
├── 📁 app/                    # Next.js app router
│   ├── capture/page.tsx       # Camera capture page
│   ├── layout.tsx            # App layout
│   └── page.tsx              # Home page
├── 📁 components/            # React components
│   ├── CameraCapture.tsx     # Main camera component
│   ├── AnalysisResults.tsx   # CV results display
│   └── Sidebar.tsx           # Navigation sidebar
├── 📁 backend/               # Python CV backend
│   ├── app.py                # Flask API server
│   ├── 📁 models/
│   │   └── person_analyzer.py # Core CV analysis
│   ├── 📁 utils/
│   │   ├── image_processor.py # Image utilities
│   │   └── response_formatter.py # API responses
│   ├── requirements.txt      # Python dependencies
│   └── README.md            # Backend documentation
├── package.json             # Node.js dependencies
├── tailwind.config.js       # Styling configuration
└── SETUP_GUIDE.md          # This file
```

---

## 🔄 **Development Workflow**

### **Making Changes**

1. **Frontend changes:**
   - Edit components in `/components/` or `/app/`
   - Hot reload automatically updates browser
   - TypeScript provides compile-time checking

2. **Backend changes:**
   - Edit Python files in `/backend/`
   - Restart Flask server to see changes
   - Test endpoints with curl or Postman

### **Adding New Features**

1. **New CV analysis capabilities:**
   - Extend `PersonAnalyzer` class
   - Update response formatter
   - Add corresponding frontend display

2. **UI improvements:**
   - Modify React components
   - Update Tailwind classes
   - Add new analysis result displays

---

## 🌟 **Advanced Usage**

### **Batch Processing**
Process multiple images programmatically:

```python
import requests
import base64
import os

def process_images_folder(folder_path):
    results = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            with open(os.path.join(folder_path, filename), 'rb') as f:
                img_data = base64.b64encode(f.read()).decode()
            
            response = requests.post(
                'http://localhost:5000/analyze-image',
                json={'image': f'data:image/jpeg;base64,{img_data}'}
            )
            results.append(response.json())
    
    return results
```

### **Real-time Video Analysis**
Use the video frame endpoint for live analysis:

```javascript
// Capture video frames at intervals
setInterval(async () => {
    if (videoRef.current) {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        canvas.width = videoRef.current.videoWidth;
        canvas.height = videoRef.current.videoHeight;
        ctx.drawImage(videoRef.current, 0, 0);
        
        const frameData = canvas.toDataURL('image/jpeg', 0.7);
        const results = await analyzeVideoFrame(frameData);
        // Process results...
    }
}, 1000); // Analyze every second
```

---

## 📈 **Performance Metrics**

### **Expected Performance:**
- **Image Analysis**: 1-3 seconds (CPU), 0.5-1 second (GPU)
- **Memory Usage**: 2-4GB RAM for model loading
- **Accuracy**: 85-95% for face detection, 80-90% for attribute analysis
- **Supported Formats**: JPEG, PNG, WebP
- **Max Image Size**: 10MB (configurable)

---

## 🆘 **Support**

### **Getting Help**
1. Check this setup guide first
2. Review console logs for error messages
3. Test individual components (frontend/backend separately)
4. Verify all dependencies are installed correctly

### **Common Commands**
```bash
# Check Python version
python --version

# Check Node.js version
node --version

# View running processes
netstat -an | findstr :3001  # Frontend
netstat -an | findstr :5000  # Backend

# Kill processes if needed
taskkill /F /IM node.exe     # Windows
killall node                # macOS/Linux
```

---

🎉 **You're now ready to capture and analyze images with advanced computer vision!** 

The system will automatically detect people, analyze their demographics, physical attributes, clothing, and poses, providing detailed insights for each captured image. 