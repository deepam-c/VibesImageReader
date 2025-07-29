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

# Import our custom modules
from models.person_analyzer import PersonAnalyzer
from utils.image_processor import ImageProcessor
from utils.response_formatter import ResponseFormatter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

# Initialize components
person_analyzer = PersonAnalyzer()
image_processor = ImageProcessor()
response_formatter = ResponseFormatter()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'service': 'Human Analysis API'
    })

@app.route('/analyze-image', methods=['POST'])
def analyze_image():
    """
    Main endpoint for analyzing human images
    Expects: JSON with base64 encoded image
    Returns: Detailed analysis of people in the image
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
            cv_image = image_processor.pil_to_opencv(image)
            
        except Exception as e:
            logger.error(f"Error decoding image: {str(e)}")
            return jsonify({'error': 'Invalid image data'}), 400
        
        # Analyze the image
        analysis_results = person_analyzer.analyze_image(cv_image)
        
        # Format response
        formatted_response = response_formatter.format_analysis(analysis_results)
        
        return jsonify(formatted_response)
        
    except Exception as e:
        logger.error(f"Error in analyze_image: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/analyze-video-frame', methods=['POST'])
def analyze_video_frame():
    """
    Endpoint for analyzing single video frames
    """
    try:
        data = request.get_json()
        
        if not data or 'frame' not in data:
            return jsonify({'error': 'No frame data provided'}), 400
        
        # Process video frame (similar to image processing)
        frame_data = data['frame']
        if ',' in frame_data:
            frame_data = frame_data.split(',')[1]
        
        frame_bytes = base64.b64decode(frame_data)
        frame_image = Image.open(io.BytesIO(frame_bytes))
        cv_frame = image_processor.pil_to_opencv(frame_image)
        
        # Analyze frame
        analysis_results = person_analyzer.analyze_image(cv_frame, is_video_frame=True)
        formatted_response = response_formatter.format_analysis(analysis_results)
        
        return jsonify(formatted_response)
        
    except Exception as e:
        logger.error(f"Error in analyze_video_frame: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/get-capabilities', methods=['GET'])
def get_capabilities():
    """
    Returns the capabilities of the analysis system
    """
    capabilities = {
        'person_detection': True,
        'face_analysis': {
            'age_estimation': True,
            'gender_detection': True,
            'emotion_recognition': True,
            'facial_landmarks': True
        },
        'body_analysis': {
            'pose_estimation': True,
            'body_parts_detection': True
        },
        'appearance_analysis': {
            'clothing_detection': True,
            'color_analysis': True,
            'style_classification': True
        },
        'hair_analysis': {
            'hair_style_detection': True,
            'hair_color_detection': True,
            'hair_length_estimation': True
        },
        'supported_formats': ['JPEG', 'PNG', 'WebP'],
        'max_image_size': '10MB',
        'processing_time': 'Real-time capable'
    }
    
    return jsonify(capabilities)

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('temp', exist_ok=True)
    os.makedirs('models/weights', exist_ok=True)
    
    # Start the Flask application
    app.run(debug=True, host='0.0.0.0', port=5000) 