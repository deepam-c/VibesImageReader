# Human Analysis API Backend

A comprehensive computer vision backend for analyzing human images and videos. This API provides detailed analysis including demographic information, physical attributes, clothing detection, pose estimation, and more.

## Features

### 🔍 **Core Analysis Capabilities**
- **Person Detection**: Locate and identify people in images
- **Face Analysis**: Age estimation, gender detection, emotion recognition
- **Body Analysis**: Pose estimation, body part detection
- **Hair Analysis**: Style, color, length, and texture detection
- **Clothing Detection**: Item identification, color analysis, style classification
- **Demographic Analysis**: Age ranges, gender distribution summaries

### 🎯 **Supported Formats**
- **Images**: JPEG, PNG, WebP
- **Input**: Base64 encoded images via REST API
- **Output**: Structured JSON responses with confidence scores

## Installation

### Prerequisites
- Python 3.8+ 
- pip package manager
- (Optional) CUDA-capable GPU for enhanced performance

### Setup Instructions

1. **Clone and Navigate**
   ```bash
   cd backend
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download Required Models**
   ```bash
   # Models will auto-download on first use
   # Ensure you have at least 2GB free space for model files
   ```

5. **Set Environment Variables (Optional)**
   ```bash
   # Create .env file
   echo "FLASK_ENV=development" > .env
   echo "LOG_LEVEL=INFO" >> .env
   ```

## Usage

### Starting the Server

```bash
python app.py
```

The server will start on `http://localhost:5000`

### API Endpoints

#### 1. Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00Z",
  "service": "Human Analysis API"
}
```

#### 2. Analyze Image
```http
POST /analyze-image
Content-Type: application/json
```

**Request Body:**
```json
{
  "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ..."
}
```

**Response:**
```json
{
  "success": true,
  "timestamp": "2024-01-01T12:00:00Z",
  "processing_info": {
    "image_dimensions": {
      "width": 1024,
      "height": 768,
      "aspect_ratio": 1.33
    },
    "processing_status": "completed"
  },
  "detection_summary": {
    "total_people_detected": 2,
    "faces_detected": 2,
    "poses_detected": 2,
    "gender_distribution": {"male": 1, "female": 1},
    "age_distribution": {"young_adult": 2}
  },
  "people": [
    {
      "person_id": 0,
      "demographics": {
        "age": {
          "estimated_age": 25,
          "age_range": "young_adult",
          "confidence": "high"
        },
        "gender": {
          "prediction": "female",
          "confidence": "high"
        }
      },
      "physical_attributes": {
        "facial_features": {
          "face_shape": "oval",
          "skin_tone": "medium"
        },
        "hair": {
          "detected": true,
          "style": "long",
          "color": "brown",
          "length": "long",
          "texture": "straight"
        },
        "body": {
          "build": "average",
          "visible_parts": ["head", "shoulders", "arms"]
        }
      },
      "appearance": {
        "clothing": {
          "detected_items": ["shirt"],
          "dominant_colors": ["blue", "white"],
          "style_category": "casual"
        },
        "overall_style": "casual"
      },
      "pose_analysis": {
        "pose_detected": true,
        "body_position": "standing",
        "activity": "standing"
      }
    }
  ]
}
```

#### 3. Analyze Video Frame
```http
POST /analyze-video-frame
Content-Type: application/json
```

Same request/response format as `/analyze-image` but optimized for video processing.

#### 4. Get Capabilities
```http
GET /get-capabilities
```

Returns detailed information about the system's analysis capabilities.

## Integration Example

### JavaScript/Node.js
```javascript
async function analyzeImage(imageBase64) {
  const response = await fetch('http://localhost:5000/analyze-image', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      image: imageBase64
    })
  });
  
  const results = await response.json();
  return results;
}
```

### Python
```python
import requests
import base64

def analyze_image(image_path):
    with open(image_path, 'rb') as img_file:
        img_base64 = base64.b64encode(img_file.read()).decode()
    
    response = requests.post(
        'http://localhost:5000/analyze-image',
        json={'image': f'data:image/jpeg;base64,{img_base64}'}
    )
    
    return response.json()
```

## Model Information

### Pre-trained Models Used
- **Face Detection**: MediaPipe Face Detection
- **Age/Gender**: DeepFace with multiple backends
- **Pose Estimation**: MediaPipe Pose
- **Object Detection**: YOLOv8 for clothing/accessories
- **Emotion Recognition**: DeepFace emotion models

### Performance Considerations
- **Processing Time**: 1-3 seconds per image (CPU), 0.5-1 second (GPU)
- **Memory Usage**: ~2-4GB RAM for model loading
- **Disk Space**: ~2GB for all model files
- **Concurrent Requests**: Supports multiple simultaneous analyses

## Configuration

### Environment Variables
```bash
FLASK_ENV=development          # Flask environment
LOG_LEVEL=INFO                # Logging level
MAX_IMAGE_SIZE=10485760       # Max image size in bytes (10MB)
ENABLE_GPU=false              # Enable GPU acceleration
MODEL_CACHE_DIR=./models      # Model storage directory
```

### Customization
- Modify confidence thresholds in `models/person_analyzer.py`
- Add custom clothing categories in the YOLO configuration
- Extend analysis capabilities by adding new model integrations

## Troubleshooting

### Common Issues

1. **Model Download Failures**
   ```bash
   # Manual model download
   python -c "from deepface import DeepFace; DeepFace.build_model('Age')"
   ```

2. **Memory Issues**
   ```bash
   # Reduce batch size or image resolution
   # Set ENABLE_GPU=false if GPU memory is limited
   ```

3. **Dependency Conflicts**
   ```bash
   # Reinstall in clean environment
   pip uninstall tensorflow deepface
   pip install -r requirements.txt
   ```

### Performance Optimization

1. **GPU Acceleration**
   ```bash
   # Install CUDA version of TensorFlow
   pip install tensorflow-gpu==2.15.0
   ```

2. **Model Caching**
   ```python
   # Models are cached after first load
   # Restart server to clear cache if needed
   ```

## Development

### Project Structure
```
backend/
├── app.py                    # Main Flask application
├── models/
│   ├── person_analyzer.py    # Core analysis logic
│   └── weights/             # Model weight files
├── utils/
│   ├── image_processor.py    # Image processing utilities
│   └── response_formatter.py # Response formatting
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

### Testing
```bash
# Run tests
pytest tests/

# Test specific endpoint
curl -X POST http://localhost:5000/health
```

### Adding New Features
1. Extend `PersonAnalyzer` class for new analysis types
2. Update `ResponseFormatter` for new response fields
3. Add corresponding API endpoints in `app.py`

## API Rate Limits
- Default: 60 requests per minute per IP
- For production use, implement Redis-based rate limiting
- Consider image size limits for optimal performance

## Security Considerations
- Validate image formats and sizes
- Implement authentication for production deployment
- Use HTTPS in production environments
- Monitor for malicious image uploads

## License
This project uses various open-source models and libraries. Please check individual model licenses for commercial usage restrictions. 