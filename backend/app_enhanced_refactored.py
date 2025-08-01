"""
Enhanced Refactored Flask application with advanced AI features
Simple version for testing without complex dependency injection
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from datetime import datetime
from services.smart_image_processor import SmartImageProcessor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_enhanced_app() -> Flask:
    """Create Flask application with enhanced AI features"""
    
    # Create Flask app
    app = Flask(__name__)
    CORS(app)
    
    # Initialize enhanced image processor
    enhanced_processor = SmartImageProcessor()
    
    @app.route('/health', methods=['GET'])
    def health_check():
        """Health check endpoint with AI status"""
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'version': '2.1.0-enhanced-ai',
            'service': 'Enhanced CV API with Advanced AI',
            'ai_backend': 'DeepFace Available' if enhanced_processor.deepface_available else 'Enhanced Mock',
            'architecture': 'Clean Architecture + Advanced AI'
        })
    
    @app.route('/analyze-image', methods=['POST'])
    async def analyze_image():
        """Enhanced image analysis endpoint"""
        try:
            data = request.get_json()
            
            if not data or 'image' not in data:
                return jsonify({'error': 'No image data provided'}), 400
            
            # Process with enhanced AI
            result = await enhanced_processor.analyze_image(data['image'])
            
            # Add analysis ID for tracking
            result['analysis_id'] = f"enhanced_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            return jsonify(result)
            
        except Exception as e:
            logger.error(f"Error in enhanced analysis: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/capabilities', methods=['GET'])
    def get_capabilities():
        """Get enhanced system capabilities"""
        try:
            capabilities = enhanced_processor.get_capabilities()
            
            # Add system information
            capabilities['system_info'] = {
                'architecture': 'Clean Architecture + Advanced AI',
                'version': '2.1.0-enhanced-ai',
                'ai_models': 'DeepFace Available' if enhanced_processor.deepface_available else 'Enhanced Mock',
                'features': [
                    'Real-time AI processing',
                    'Age and gender detection',
                    'Emotion recognition',
                    'Advanced face detection',
                    'Hair and color analysis',
                    'Scene analysis',
                    'Multi-level confidence scoring'
                ]
            }
            
            return jsonify(capabilities)
            
        except Exception as e:
            logger.error(f"Error getting capabilities: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/test-ai', methods=['POST'])
    async def test_ai():
        """Simple AI test endpoint"""
        try:
            # Create a simple test image (red square)
            import numpy as np
            import cv2
            import base64
            from PIL import Image
            import io
            
            # Generate test image
            test_img = np.full((200, 200, 3), [0, 0, 255], dtype=np.uint8)  # Red image
            
            # Convert to base64
            pil_img = Image.fromarray(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB))
            buffer = io.BytesIO()
            pil_img.save(buffer, format='JPEG')
            img_bytes = buffer.getvalue()
            base64_string = base64.b64encode(img_bytes).decode('utf-8')
            
            # Analyze
            result = await enhanced_processor.analyze_image(f"data:image/jpeg;base64,{base64_string}")
            
            return jsonify({
                'test_status': 'success',
                'ai_available': enhanced_processor.deepface_available,
                'analysis_summary': {
                    'people_detected': result.get('detection_summary', {}).get('total_people_detected', 0),
                    'processing_time': result.get('processing_info', {}).get('processing_time_ms', 0),
                    'confidence': result.get('processing_info', {}).get('overall_confidence', 0)
                }
            })
            
        except Exception as e:
            logger.error(f"Error in AI test: {e}")
            return jsonify({'test_status': 'error', 'error': str(e)}), 500
    
    logger.info("🚀 Enhanced CV API created successfully")
    return app

if __name__ == '__main__':
    app = create_enhanced_app()
    
    print("🚀 Starting Enhanced Computer Vision API...")
    print("🤖 AI Models: Advanced AI Features Integrated")
    print("🏗️ Architecture: Clean Design + Enhanced AI")
    print("✅ Server starting on http://localhost:5000")
    print("\n📍 Available Endpoints:")
    print("  GET  /health       - Health check with AI status")
    print("  POST /analyze-image - Enhanced AI image analysis")
    print("  GET  /capabilities  - System capabilities")
    print("  POST /test-ai       - Simple AI functionality test")
    
    app.run(host='0.0.0.0', port=5000, debug=True) 