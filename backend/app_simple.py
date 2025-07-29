from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import cv2
import numpy as np
from PIL import Image
import io
import logging
from datetime import datetime
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'service': 'Human Analysis API (Simplified)'
    })

@app.route('/analyze-image', methods=['POST'])
def analyze_image():
    """
    Simplified image analysis endpoint
    """
    try:
        # Get the JSON data
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Decode base64 image
        try:
            image_data = data['image']
            # Remove data URL prefix if present
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            
            # Decode base64
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to OpenCV format
            cv_image = pil_to_opencv(image)
            
        except Exception as e:
            logger.error(f"Error decoding image: {str(e)}")
            return jsonify({'error': 'Invalid image data'}), 400
        
        # Basic image analysis
        analysis_results = analyze_image_basic(cv_image)
        
        return jsonify(analysis_results)
        
    except Exception as e:
        logger.error(f"Error in analyze_image: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/get-capabilities', methods=['GET'])
def get_capabilities():
    """
    Returns the capabilities of the simplified analysis system
    """
    capabilities = {
        'person_detection': 'Basic edge detection',
        'face_analysis': {
            'age_estimation': False,
            'gender_detection': False,
            'emotion_recognition': False,
            'facial_landmarks': False
        },
        'body_analysis': {
            'pose_estimation': False,
            'body_parts_detection': False
        },
        'appearance_analysis': {
            'clothing_detection': False,
            'color_analysis': True,
            'style_classification': False
        },
        'hair_analysis': {
            'hair_style_detection': False,
            'hair_color_detection': 'Basic',
            'hair_length_estimation': False
        },
        'supported_formats': ['JPEG', 'PNG', 'WebP'],
        'max_image_size': '10MB',
        'processing_time': 'Real-time',
        'note': 'Simplified version - install full AI models for advanced features'
    }
    
    return jsonify(capabilities)

def pil_to_opencv(pil_image: Image.Image) -> np.ndarray:
    """Convert PIL Image to OpenCV format (BGR)"""
    try:
        # Convert PIL image to RGB if not already
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Convert to numpy array
        np_array = np.array(pil_image)
        
        # Convert RGB to BGR for OpenCV
        cv_image = cv2.cvtColor(np_array, cv2.COLOR_RGB2BGR)
        
        return cv_image
        
    except Exception as e:
        logger.error(f"Error converting PIL to OpenCV: {e}")
        raise

def analyze_image_basic(image: np.ndarray) -> dict:
    """Basic image analysis without heavy AI models"""
    
    # Get image info
    height, width = image.shape[:2]
    
    # Basic color analysis
    dominant_colors = get_dominant_colors(image)
    
    # Improved person detection using multiple methods
    person_detected, face_detected = detect_person_and_face(image)
    
    # Calculate edge percentage for additional analysis
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 100)  # Lower thresholds for better detection
    edge_percentage = (np.sum(edges > 0) / (height * width)) * 100
    
    # Mock analysis results that match the expected format
    results = {
        'success': True,
        'timestamp': datetime.now().isoformat(),
        'processing_info': {
            'image_dimensions': {
                'width': int(width),
                'height': int(height),
                'aspect_ratio': round(width / height, 2)
            },
            'processing_status': 'completed',
            'analysis_version': '1.0.0-simplified'
        },
                 'detection_summary': {
             'total_people_detected': 1 if person_detected else 0,
             'faces_detected': 1 if face_detected else 0,
             'poses_detected': 1 if person_detected else 0,  # Assume pose if person detected
             'gender_distribution': {'unknown': 1 if person_detected else 0},
             'age_distribution': {'unknown': 1 if person_detected else 0},
            'confidence_scores': {
                'overall': 0.3,  # Low confidence for basic analysis
                'face_detection': 0.0,
                'pose_detection': 0.0
            }
        },
        'people': [],
                 'scene_analysis': {
             'scene_type': 'portrait' if person_detected else 'no_people',
             'lighting_conditions': assess_lighting(image),
             'image_quality': assess_image_quality(width, height),
             'analysis_challenges': ['Advanced AI models not loaded'] if not person_detected else ['Using basic detection methods']
         },
        'metadata': {
            'analysis_timestamp': datetime.now().isoformat(),
            'processing_time_ms': 'Fast (<100ms)',
            'model_versions': {
                'face_detection': 'Not available',
                'age_gender': 'Not available',
                'pose_estimation': 'Not available',
                'object_detection': 'Not available'
            },
            'capabilities_used': ['basic_image_processing', 'color_analysis']
        }
    }
    
    # Add a mock person if person detected
    if person_detected:
        mock_person = {
            'person_id': 0,
            'demographics': {
                'age': {
                    'estimated_age': 'unknown',
                    'age_range': 'unknown',
                    'confidence': 'low'
                },
                'gender': {
                    'prediction': 'unknown',
                    'confidence': 'very_low'
                }
            },
            'physical_attributes': {
                'facial_features': {
                    'face_shape': 'unknown',
                    'skin_tone': 'unknown',
                    'facial_hair': 'unknown'
                },
                'hair': {
                    'detected': False,
                    'style': 'unknown',
                    'color': dominant_colors[0] if dominant_colors else 'unknown',
                    'length': 'unknown',
                    'texture': 'unknown'
                },
                'body': {
                    'build': 'unknown',
                    'height_estimate': 'unknown',
                    'visible_parts': ['unknown']
                }
            },
            'appearance': {
                'clothing': {
                    'detected_items': [],
                    'dominant_colors': dominant_colors,
                    'style_category': 'unknown'
                },
                'accessories': [],
                'overall_style': 'unknown',
                'color_palette': dominant_colors
            },
            'pose_analysis': {
                'pose_detected': False,
                'body_position': 'unknown',
                'pose_confidence': 0,
                'activity': 'unknown',
                'orientation': 'unknown'
            },
                         'confidence_metrics': {
                 'overall_confidence': 0.7 if face_detected else 0.5,
                 'face_detection_confidence': 0.8 if face_detected else 0,
                 'age_confidence': 'very_low',
                 'gender_confidence': 'very_low',
                 'pose_confidence': 0.6 if person_detected else 0
             }
        }
        results['people'].append(mock_person)
    
    return results

def get_dominant_colors(image: np.ndarray, num_colors: int = 3) -> list:
    """Extract dominant colors from image"""
    try:
        # Resize image for faster processing
        small_image = cv2.resize(image, (50, 50))
        
        # Reshape to 1D array of pixels
        data = small_image.reshape((-1, 3))
        data = np.float32(data)
        
        # Use K-means to find dominant colors
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(data, num_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Convert to color names
        colors = []
        for center in centers:
            color_name = rgb_to_color_name(center)
            colors.append(color_name)
        
        return colors
    except:
        return ['unknown']

def rgb_to_color_name(rgb: np.ndarray) -> str:
    """Convert RGB to basic color name"""
    r, g, b = rgb
    
    if r > 200 and g > 200 and b > 200:
        return 'white'
    elif r < 50 and g < 50 and b < 50:
        return 'black'
    elif r > max(g, b) + 50:
        return 'red'
    elif g > max(r, b) + 50:
        return 'green'
    elif b > max(r, g) + 50:
        return 'blue'
    elif r > 150 and g > 150 and b < 100:
        return 'yellow'
    elif r > 150 and g < 100 and b > 150:
        return 'purple'
    elif r > 200 and g > 100 and b < 100:
        return 'orange'
    else:
        return 'mixed'

def assess_lighting(image: np.ndarray) -> str:
    """Assess lighting conditions"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    
    if mean_brightness < 80:
        return 'low'
    elif mean_brightness > 180:
        return 'bright'
    else:
        return 'adequate'

def detect_person_and_face(image: np.ndarray) -> tuple:
    """Detect person and face using OpenCV built-in methods"""
    person_detected = False
    face_detected = False
    
    try:
        # Try to load face detection cascade
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))
        
        if len(faces) > 0:
            face_detected = True
            person_detected = True  # If face detected, person is definitely present
            logger.info(f"Detected {len(faces)} face(s)")
        else:
            # If no faces, use improved edge detection and contour analysis
            person_detected = detect_person_by_contours(image)
            
    except Exception as e:
        logger.warning(f"Face detection failed: {e}")
        # Fallback to contour-based detection
        person_detected = detect_person_by_contours(image)
    
    return person_detected, face_detected

def detect_person_by_contours(image: np.ndarray) -> bool:
    """Detect person using contour analysis and shape detection"""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection with better parameters
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        height, width = image.shape[:2]
        image_area = height * width
        
        # Look for significant contours that could be a person
        for contour in contours:
            area = cv2.contourArea(contour)
            # If contour area is between 5% and 80% of image, likely a person
            if 0.05 * image_area < area < 0.8 * image_area:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = h / w if w > 0 else 0
                
                # Human-like aspect ratio (taller than wide, typically 1.5-3.0)
                if 1.2 < aspect_ratio < 4.0:
                    # Check if it's vertically oriented and reasonably sized
                    if h > height * 0.3 and w > width * 0.1:
                        logger.info(f"Person-like contour detected: area={area}, aspect_ratio={aspect_ratio}")
                        return True
        
        # Additional check: analyze skin-tone colors for face regions
        return detect_skin_tone_regions(image)
        
    except Exception as e:
        logger.error(f"Contour detection failed: {e}")
        return False

def detect_skin_tone_regions(image: np.ndarray) -> bool:
    """Detect skin-tone regions as indicator of human presence"""
    try:
        # Convert to HSV for better skin detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define skin color range in HSV
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        # Create mask for skin colors
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Calculate percentage of skin-colored pixels
        skin_pixels = np.sum(mask > 0)
        total_pixels = mask.shape[0] * mask.shape[1]
        skin_percentage = (skin_pixels / total_pixels) * 100
        
        # If significant skin-tone area detected, likely a person
        if skin_percentage > 8:  # At least 8% skin-tone pixels
            logger.info(f"Skin-tone regions detected: {skin_percentage:.1f}%")
            return True
            
        return False
        
    except Exception as e:
        logger.error(f"Skin tone detection failed: {e}")
        return False

def assess_image_quality(width: int, height: int) -> str:
    """Assess image quality based on resolution"""
    if width >= 1024 and height >= 768:
        return 'high'
    elif width >= 640 and height >= 480:
        return 'medium'
    else:
        return 'low'

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('temp', exist_ok=True)
    
    print("🚀 Starting simplified Computer Vision API...")
    print("📝 Note: This is a basic version. Install full AI models for advanced features.")
    print("✅ Server starting on http://localhost:5000")
    
    # Start the Flask application
    app.run(debug=True, host='0.0.0.0', port=5000) 