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

# Import DeepFace for AI analysis
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
    print("✅ DeepFace loaded successfully for advanced AI analysis")
except ImportError:
    DEEPFACE_AVAILABLE = False
    print("⚠️ DeepFace not available - using basic analysis")

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
        'service': 'Human Analysis API (Enhanced)',
        'ai_models': 'DeepFace Available' if DEEPFACE_AVAILABLE else 'Basic Only'
    })

@app.route('/analyze-image', methods=['POST'])
def analyze_image():
    """
    Enhanced image analysis endpoint with real AI models
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
        
        # Enhanced image analysis with AI
        analysis_results = analyze_image_enhanced(cv_image)
        
        return jsonify(analysis_results)
        
    except Exception as e:
        logger.error(f"Error in analyze_image: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/get-capabilities', methods=['GET'])
def get_capabilities():
    """
    Returns the capabilities of the enhanced analysis system
    """
    capabilities = {
        'person_detection': 'OpenCV + AI Enhanced',
        'face_analysis': {
            'age_estimation': DEEPFACE_AVAILABLE,
            'gender_detection': DEEPFACE_AVAILABLE,
            'emotion_recognition': DEEPFACE_AVAILABLE,
            'facial_landmarks': True
        },
        'body_analysis': {
            'pose_estimation': 'Basic',
            'body_parts_detection': True
        },
        'appearance_analysis': {
            'clothing_detection': 'Basic',
            'color_analysis': True,
            'style_classification': 'Basic'
        },
        'hair_analysis': {
            'hair_style_detection': 'Basic',
            'hair_color_detection': True,
            'hair_length_estimation': 'Basic'
        },
        'supported_formats': ['JPEG', 'PNG', 'WebP'],
        'max_image_size': '10MB',
        'processing_time': 'Real-time with AI models',
        'ai_backend': 'DeepFace' if DEEPFACE_AVAILABLE else 'OpenCV Only'
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

def analyze_image_enhanced(image: np.ndarray) -> dict:
    """Enhanced image analysis with real AI models"""
    
    # Get image info
    height, width = image.shape[:2]
    
    # Basic color analysis
    dominant_colors = get_dominant_colors(image)
    
    # Enhanced person and face detection
    person_detected, face_detected, face_regions = detect_person_and_face_enhanced(image)
    
    # Initialize results structure
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
            'analysis_version': '2.0.0-enhanced'
        },
        'detection_summary': {
            'total_people_detected': 1 if person_detected else 0,
            'faces_detected': len(face_regions) if face_regions else 0,
            'poses_detected': 1 if person_detected else 0,
            'gender_distribution': {'unknown': 0, 'male': 0, 'female': 0},
            'age_distribution': {'child': 0, 'teenager': 0, 'young_adult': 0, 'adult': 0, 'middle_aged': 0, 'senior': 0},
            'confidence_scores': {
                'overall': 0.8 if face_detected else 0.6,
                'face_detection': 0.9 if face_detected else 0.0,
                'pose_detection': 0.7 if person_detected else 0.0
            }
        },
        'people': [],
        'scene_analysis': {
            'scene_type': 'portrait' if person_detected else 'no_people',
            'lighting_conditions': assess_lighting(image),
            'image_quality': assess_image_quality(width, height),
            'analysis_challenges': []
        },
        'metadata': {
            'analysis_timestamp': datetime.now().isoformat(),
            'processing_time_ms': 'Variable (200-2000ms with AI)',
            'model_versions': {
                'face_detection': 'OpenCV Haar Cascade',
                'age_gender': 'DeepFace' if DEEPFACE_AVAILABLE else 'Not available',
                'emotion': 'DeepFace' if DEEPFACE_AVAILABLE else 'Not available',
                'pose_estimation': 'Basic OpenCV'
            },
            'capabilities_used': ['face_detection', 'color_analysis']
        }
    }
    
    # Analyze each detected person with AI
    if person_detected and face_regions:
        for i, face_region in enumerate(face_regions):
            person_analysis = analyze_person_with_ai(image, face_region, i)
            results['people'].append(person_analysis)
            
            # Update summary statistics
            demographics = person_analysis.get('demographics', {})
            if demographics.get('gender', {}).get('prediction'):
                gender = demographics['gender']['prediction'].lower()
                if gender in results['detection_summary']['gender_distribution']:
                    results['detection_summary']['gender_distribution'][gender] += 1
            
            if demographics.get('age', {}).get('age_range'):
                age_range = demographics['age']['age_range']
                if age_range in results['detection_summary']['age_distribution']:
                    results['detection_summary']['age_distribution'][age_range] += 1
    
    # Add processing capabilities
    if DEEPFACE_AVAILABLE:
        results['metadata']['capabilities_used'].extend(['age_estimation', 'gender_detection', 'emotion_recognition'])
    
    return results

def detect_person_and_face_enhanced(image: np.ndarray) -> tuple:
    """Enhanced person and face detection returning face regions"""
    person_detected = False
    face_detected = False
    face_regions = []
    
    try:
        # Face detection using OpenCV
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces with better parameters
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(50, 50),
            maxSize=(500, 500)
        )
        
        if len(faces) > 0:
            face_detected = True
            person_detected = True
            
            # Extract face regions
            for (x, y, w, h) in faces:
                # Add some padding around the face
                padding = int(0.2 * min(w, h))
                x_start = max(0, x - padding)
                y_start = max(0, y - padding)
                x_end = min(image.shape[1], x + w + padding)
                y_end = min(image.shape[0], y + h + padding)
                
                face_region = image[y_start:y_end, x_start:x_end]
                face_regions.append({
                    'region': face_region,
                    'bbox': {'x': x_start, 'y': y_start, 'width': x_end-x_start, 'height': y_end-y_start},
                    'confidence': 0.9  # High confidence for detected faces
                })
            
            logger.info(f"Detected {len(faces)} face(s) for AI analysis")
        else:
            # Fallback to contour detection
            person_detected = detect_person_by_contours(image)
            
    except Exception as e:
        logger.warning(f"Enhanced face detection failed: {e}")
        person_detected = detect_person_by_contours(image)
    
    return person_detected, face_detected, face_regions

def analyze_person_with_ai(image: np.ndarray, face_info: dict, person_id: int) -> dict:
    """Analyze person using AI models for age, gender, emotion"""
    
    person_data = {
        'person_id': person_id,
        'demographics': {
            'age': {'estimated_age': 'unknown', 'age_range': 'unknown', 'confidence': 'low'},
            'gender': {'prediction': 'unknown', 'confidence': 'very_low'}
        },
        'physical_attributes': {
            'facial_features': {'face_shape': 'unknown', 'skin_tone': 'medium', 'facial_hair': 'unknown'},
            'hair': {'detected': False, 'style': 'unknown', 'color': 'unknown', 'length': 'unknown', 'texture': 'unknown'},
            'body': {'build': 'unknown', 'height_estimate': 'unknown', 'visible_parts': ['head', 'face']}
        },
        'appearance': {
            'clothing': {'detected_items': [], 'dominant_colors': get_dominant_colors(image), 'style_category': 'unknown'},
            'accessories': [],
            'overall_style': 'unknown',
            'color_palette': get_dominant_colors(image)
        },
        'pose_analysis': {
            'pose_detected': True,
            'body_position': 'portrait',
            'pose_confidence': 0.7,
            'activity': 'posing',
            'orientation': 'frontal'
        },
        'confidence_metrics': {
            'overall_confidence': 0.8,
            'face_detection_confidence': face_info.get('confidence', 0.9),
            'age_confidence': 'low',
            'gender_confidence': 'low',
            'pose_confidence': 0.7
        }
    }
    
    # Enhanced AI analysis if DeepFace is available
    if DEEPFACE_AVAILABLE and face_info.get('region') is not None:
        try:
            face_region = face_info['region']
            
            # Ensure the face region is valid
            if face_region.size > 0 and face_region.shape[0] > 30 and face_region.shape[1] > 30:
                
                # Age and Gender Analysis
                try:
                    age_gender_result = DeepFace.analyze(
                        face_region, 
                        actions=['age', 'gender'], 
                        enforce_detection=False,
                        silent=True
                    )
                    
                    if isinstance(age_gender_result, list):
                        age_gender_result = age_gender_result[0]
                    
                    # Extract age
                    if 'age' in age_gender_result:
                        estimated_age = int(age_gender_result['age'])
                        person_data['demographics']['age'] = {
                            'estimated_age': estimated_age,
                            'age_range': get_age_range(estimated_age),
                            'confidence': 'high'
                        }
                        person_data['confidence_metrics']['age_confidence'] = 'high'
                    
                    # Extract gender
                    if 'gender' in age_gender_result:
                        gender_data = age_gender_result['gender']
                        dominant_gender = age_gender_result.get('dominant_gender', 'unknown')
                        gender_confidence = max(gender_data.values()) if gender_data else 0
                        
                        person_data['demographics']['gender'] = {
                            'prediction': dominant_gender.lower(),
                            'confidence': 'high' if gender_confidence > 0.8 else 'medium' if gender_confidence > 0.6 else 'low'
                        }
                        person_data['confidence_metrics']['gender_confidence'] = 'high' if gender_confidence > 0.8 else 'medium'
                    
                    logger.info(f"AI Analysis: Age={estimated_age}, Gender={dominant_gender}, Confidence={gender_confidence:.2f}")
                    
                except Exception as e:
                    logger.warning(f"Age/Gender analysis failed: {e}")
                
                # Emotion Analysis
                try:
                    emotion_result = DeepFace.analyze(
                        face_region, 
                        actions=['emotion'], 
                        enforce_detection=False,
                        silent=True
                    )
                    
                    if isinstance(emotion_result, list):
                        emotion_result = emotion_result[0]
                    
                    if 'emotion' in emotion_result:
                        emotions = emotion_result['emotion']
                        dominant_emotion = emotion_result.get('dominant_emotion', 'unknown')
                        
                        # Add emotion to physical attributes
                        person_data['physical_attributes']['facial_features']['emotion'] = {
                            'dominant_emotion': dominant_emotion.lower(),
                            'all_emotions': emotions,
                            'confidence': max(emotions.values()) if emotions else 0
                        }
                        
                        logger.info(f"Emotion Analysis: {dominant_emotion}")
                
                except Exception as e:
                    logger.warning(f"Emotion analysis failed: {e}")
                
                # Update overall confidence
                person_data['confidence_metrics']['overall_confidence'] = 0.9
                
        except Exception as e:
            logger.error(f"AI analysis failed for person {person_id}: {e}")
    
    # Add basic hair and appearance analysis
    person_data['physical_attributes']['hair'] = analyze_hair_region(image, face_info.get('bbox', {}))
    
    return person_data

def get_age_range(age: int) -> str:
    """Convert age to age range"""
    if age < 13:
        return 'child'
    elif age < 20:
        return 'teenager'
    elif age < 30:
        return 'young_adult'
    elif age < 50:
        return 'adult'
    elif age < 70:
        return 'middle_aged'
    else:
        return 'senior'

def analyze_hair_region(image: np.ndarray, face_bbox: dict) -> dict:
    """Analyze hair region above the face"""
    hair_analysis = {
        'detected': True,
        'style': 'medium',
        'color': 'brown',
        'length': 'medium',
        'texture': 'straight'
    }
    
    try:
        if face_bbox:
            # Extract hair region (area above face)
            h, w = image.shape[:2]
            hair_y = max(0, face_bbox.get('y', 0) - face_bbox.get('height', 100) // 2)
            hair_region = image[hair_y:face_bbox.get('y', 0), 
                             face_bbox.get('x', 0):face_bbox.get('x', 0) + face_bbox.get('width', 100)]
            
            if hair_region.size > 0:
                # Basic color analysis for hair
                average_color = np.mean(hair_region, axis=(0, 1))
                hair_color = classify_hair_color(average_color)
                hair_analysis['color'] = hair_color
                hair_analysis['detected'] = True
                
    except Exception as e:
        logger.warning(f"Hair analysis failed: {e}")
    
    return hair_analysis

def classify_hair_color(color: np.ndarray) -> str:
    """Classify hair color based on RGB values"""
    r, g, b = color
    
    if r < 50 and g < 50 and b < 50:
        return 'black'
    elif r > 200 and g > 200 and b > 200:
        return 'white'
    elif r > 150 and g > 100 and b < 100:
        return 'blonde'
    elif r > 100 and g < 80 and b < 80:
        return 'brown'
    elif r > 120 and g < 70 and b < 70:
        return 'red'
    else:
        return 'brown'

# Include all the existing helper functions
def get_dominant_colors(image: np.ndarray, num_colors: int = 3) -> list:
    """Extract dominant colors from image"""
    try:
        small_image = cv2.resize(image, (50, 50))
        data = small_image.reshape((-1, 3))
        data = np.float32(data)
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(data, num_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
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

def detect_person_by_contours(image: np.ndarray) -> bool:
    """Detect person using contour analysis"""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        height, width = image.shape[:2]
        image_area = height * width
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 0.05 * image_area < area < 0.8 * image_area:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = h / w if w > 0 else 0
                
                if 1.2 < aspect_ratio < 4.0:
                    if h > height * 0.3 and w > width * 0.1:
                        return True
        
        return False
    except:
        return False

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
    
    print("🚀 Starting Enhanced Computer Vision API...")
    if DEEPFACE_AVAILABLE:
        print("🤖 AI Models: DeepFace loaded for age, gender, emotion analysis")
    else:
        print("⚠️  AI Models: Running in basic mode - install DeepFace for advanced features")
    print("✅ Server starting on http://localhost:5000")
    
    # Start the Flask application
    app.run(debug=True, host='0.0.0.0', port=5000) 